// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        cancel_cart
// @match       https://www.newworld.co.nz/*
// @description Empty the New World shopping cart, removing every item in one
// @description go. This cannot be undone and there is no confirmation step, so
// @description only call it when the shopper has clearly asked to start the
// @description shop over or abandon the basket. To remove single items
// @description instead, use update_cart with a quantity of 0.
// @schema      {"type":"object","properties":{}}
// ==/WebMCP==

// Emptying is DELETE /cart on api-prod.newworld.co.nz, which discards the whole
// basket for the session. Because that is irreversible, read the cart first and
// list what was thrown away: if the shopper says "no, not that" a moment later,
// the tool result is the only remaining record of what was in there.
//
// The endpoint answers 204 as readily as 200 depending on whether a cart
// existed, so both count as success.

const API = 'https://api-prod.newworld.co.nz/v1/edge';

// This route exchanges the HttpOnly session cookies for a short-lived bearer
// token, and provisions an anonymous session when there is no session at all,
// so it covers a first-time visitor on its own. There is deliberately no
// fallback to /next/api/user/login/guest: that mints a *new* session and
// rewrites refresh_token, so reaching for it when this call fails transiently
// would sign a logged-in shopper out and strand the cart they had going.
const getToken = async () => {
  const response = await fetch('/next/api/user/get-current-user', {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
  if (!response.ok) {
    throw new Error('the site would not issue a session token (HTTP ' +
      response.status + ')');
  }
  const data = await response.json().catch(() => null);
  if (!data || !data.access_token) {
    throw new Error('the site returned no session token');
  }
  return data.access_token;
};

const money = (cents) => '$' + (Number(cents || 0) / 100).toFixed(2);

// WEIGHT lines are ordered in whole grams. The Number round-trip drops the
// trailing zeros toFixed adds without touching significant ones, so 10000g
// reads as 10kg rather than 1kg.
const quantityText = (line) => {
  if (line.sale_type !== 'WEIGHT') {
    return 'x' + line.quantity;
  }
  const grams = Number(line.quantity) || 0;
  return grams >= 1000
    ? String(Number((grams / 1000).toFixed(3))) + 'kg' : grams + 'g';
};

try {
  const token = await getToken();
  const headers = {
    Accept: 'application/json',
    'Content-Type': 'application/json',
    Authorization: 'Bearer ' + token,
  };

  const readResponse = await fetch(API + '/cart', {
    credentials: 'include',
    headers: headers,
  });
  const cart = readResponse.ok ? ((await readResponse.json()) || {}) : {};
  const products = cart.products || [];

  if (!products.length) {
    return 'The New World cart is already empty. Nothing was removed.';
  }

  const emptied = await fetch(API + '/cart', {
    method: 'DELETE',
    credentials: 'include',
    headers: headers,
  });
  if (emptied.status !== 200 && emptied.status !== 204) {
    return 'Could not empty the cart: HTTP ' + emptied.status +
      '. Nothing was removed.';
  }

  const lines = [];
  lines.push('Emptied the New World cart' +
    ((cart.store || {}).storeName ? ' at ' + cart.store.storeName : '') +
    '. Removed ' + products.length + ' line' +
    (products.length === 1 ? '' : 's') + ' worth ' + money(cart.subtotal) +
    ':');
  products.forEach((line) => {
    lines.push('- ' + [line.brand, line.name].filter(Boolean).join(' ') + ' ' +
      quantityText(line) + ' (' + line.productId + ')');
  });
  lines.push('');
  lines.push('This cannot be undone, but the ids above can be passed back to ' +
    'update_cart to rebuild the basket.');
  return lines.join('\n');
} catch (error) {
  return 'Could not empty the New World cart: ' +
    (error && error.message ? error.message : String(error));
}
