// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        get_cart
// @match       https://www.newworld.co.nz/*
// @description Read the current New World shopping cart: which store it is
// @description for, every line with its quantity and line price, anything that
// @description has become unavailable, the bag and service fees, and the
// @description estimated total. Read this before changing quantities with
// @description update_cart, because update_cart sets quantities outright
// @description rather than adding to them.
// @schema      {"type":"object","properties":{}}
// ==/WebMCP==

// The cart lives server-side against the session, not in the DOM: the header
// count is client-rendered from GET /cart on api-prod.newworld.co.nz and the
// full contents only render on the cart page. Read the API so the tool works
// from anywhere on the site.
//
// Money is in cents throughout. Line "price" is already the extended price for
// the quantity, not a unit price. Quantities are pieces for UNITS lines but
// grams for WEIGHT lines, which is why they are formatted differently below.

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

const plural = (count, noun) => count + ' ' + noun + (count === 1 ? '' : 's');

// WEIGHT lines are ordered in whole grams; show them the way a shopper thinks.
// Round-tripping through Number drops the trailing zeros toFixed adds without
// touching significant ones, so 10000g stays 10kg rather than becoming 1kg.
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
  const response = await fetch(API + '/cart', {
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      Authorization: 'Bearer ' + token,
    },
  });
  if (!response.ok) {
    throw new Error('GET /cart returned HTTP ' + response.status);
  }
  const cart = (await response.json()) || {};

  const store = cart.store || {};
  const products = cart.products || [];
  const lines = [];

  if (store.storeName) {
    lines.push('Cart for ' + store.storeName +
      (store.storeAddress ? ' (' + store.storeAddress + ')' : '') + '.');
  } else {
    lines.push('Cart is not yet tied to a store. Use change_store to pick ' +
      'one before adding anything.');
  }

  if (cart.orderNumber) {
    lines.push('This cart is an edit in progress of already-placed order ' +
      cart.orderNumber + '. Changes here alter that order.');
  }

  if (!products.length) {
    lines.push('');
    lines.push('The cart is empty.');
    return lines.join('\n');
  }

  // Weight lines are a single "item" no matter how many grams, matching the
  // count the site shows in its header.
  const itemCount = products.reduce((total, line) => total +
    (line.sale_type === 'WEIGHT' ? 1 : (Number(line.quantity) || 0)), 0);
  lines.push(plural(products.length, 'line') + ', ' +
    plural(itemCount, 'item') + '.');
  lines.push('');

  products.forEach((line, index) => {
    const title = [line.brand, line.name].filter(Boolean).join(' ');
    lines.push((index + 1) + '. ' + title + ' ' + quantityText(line) + ' - ' +
      money(line.price));
    const notes = [];
    if (line.isLiquor || line.isTobacco) {
      notes.push('age restricted');
    }
    if (line.allowSubstitutions === false) {
      notes.push('no substitutions');
    }
    if (line.comment) {
      notes.push('note: ' + line.comment);
    }
    lines.push('   id ' + line.productId +
      (notes.length ? ' - ' + notes.join(', ') : ''));
  });

  const unavailable = cart.unavailableProducts || [];
  if (unavailable.length) {
    lines.push('');
    lines.push('No longer available at this store, and excluded from the ' +
      'total:');
    unavailable.forEach((line) => {
      lines.push('- ' + [line.brand, line.name].filter(Boolean).join(' ') +
        ' (' + line.productId + ')');
    });
  }

  lines.push('');
  lines.push('Subtotal: ' + money(cart.subtotal));
  if (cart.bagFee) {
    lines.push('Bag fee: ' + money(cart.bagFee));
  }
  if (cart.serviceFee) {
    lines.push('Service fee: ' + money(cart.serviceFee));
  }
  if (cart.promoCodeDiscount) {
    lines.push('Promo code discount: ' + money(cart.promoCodeDiscount));
  }
  if (cart.subscriptionDiscount) {
    lines.push('Delivery pass discount: ' + money(cart.subscriptionDiscount));
  }
  // Mirror the site's own arithmetic: the discounts arrive already signed, so
  // they are added rather than subtracted.
  const fees = (Number(cart.serviceFee) || 0) + (Number(cart.bagFee) || 0);
  const discounts = (Number(cart.promoCodeDiscount) || 0) +
    (Number(cart.subscriptionDiscount) || 0);
  lines.push('Estimated total: ' +
    money(fees + (Number(cart.subtotal) || 0) + discounts));
  lines.push('Excludes any delivery or collection fee, which depends on the ' +
    'timeslot chosen at checkout.');

  if (cart.whenLastPriced) {
    lines.push('Priced at ' + cart.whenLastPriced + '.');
  }
  if (cart.clubMember === false) {
    lines.push('Not signed in as a Clubcard member, so Club Deal prices are ' +
      'not applied.');
  }

  return lines.join('\n');
} catch (error) {
  return 'Could not read the New World cart: ' +
    (error && error.message ? error.message : String(error));
}
