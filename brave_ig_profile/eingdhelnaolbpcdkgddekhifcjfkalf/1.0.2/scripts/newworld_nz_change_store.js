// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        change_store
// @match       https://www.newworld.co.nz/*
// @description Switch the New World store that the shop is ordering from, by
// @description store name such as "Thorndon" or "New World Greymouth". Prices,
// @description availability and the whole catalogue are per-store, so do this
// @description before searching or adding to the cart if the shopper wants a
// @description different store. Call with no arguments to report the store that
// @description is currently selected and list the alternatives.
// @schema      {"type":"object","properties":{"store":{"type":"string","description":"Store name or a distinctive part of one, for example \"Greymouth\", \"New World Thorndon\" or \"Vic Park\". Omit to report the currently selected store instead of changing it."}}}
// ==/WebMCP==

// newworld.co.nz is a Next.js app whose store picker is a client-rendered modal
// backed by a map, so the store list is not dependably in the DOM. Everything
// here goes through the same API the page itself uses:
// api-prod.newworld.co.nz/v1/edge, authorised with a short-lived bearer token.
//
// That token lives in the HttpOnly "fs-user-token" cookie, which script cannot
// read. The page works around this by POSTing to its own same-origin BFF route,
// which exchanges the HttpOnly session/refresh cookies for a JWT it can put in
// an Authorization header. We do exactly the same.
//
// Changing store is POST /cart/store/<id>. It is a cart operation, not a
// preference: the server reprices the basket against the new store and answers
// 412 when something in the cart is not sold there. We surface those items and
// stop, rather than silently discarding them the way the site's own modal
// offers to.

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

const readCookie = (name) => {
  const match = document.cookie.match(
    new RegExp('(?:^|; )' + name + '=([^;]*)'));
  return match ? decodeURIComponent(match[1]) : '';
};

// The site stores its selection in "STORE_ID_V2" as "<uuid>|<isInitialStore>",
// and mirrors the bare id into "eCom_STORE_ID" for its older code paths.
const writeCookie = (name, value) => {
  const expires = new Date(Date.now() + 365 * 86400000).toUTCString();
  document.cookie = name + '=' + encodeURIComponent(value) +
    '; expires=' + expires + '; path=/; secure';
};

const currentStoreId = () =>
  readCookie('STORE_ID_V2').split('|')[0] || readCookie('eCom_STORE_ID');

const normalise = (value) =>
  String(value === undefined || value === null ? '' : value)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

const ISLANDS = { NI: 'North Island', SI: 'South Island' };

// Opening hours arrive as seven entries keyed by weekday name. Report only
// today's, so the answer stays short and matches what the shopper cares about.
const todayHours = (store) => {
  const days = ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY',
    'FRIDAY', 'SATURDAY'];
  const today = days[new Date().getDay()];
  const entry = (store.openingHours || []).find((day) => day.day === today);
  return entry ? entry.open + ' to ' + entry.close : 'unknown';
};

const describe = (store) => {
  const lines = [];
  lines.push(store.name + ' (' + store.address + ')');
  const methods = [];
  if (store.delivery || store.supportsDelivery) {
    methods.push('delivery');
  }
  if (store.clickAndCollect || store.supportsClickAndCollect) {
    methods.push('click and collect');
  }
  lines.push('  Region: ' + (ISLANDS[store.region] || store.region ||
    'unknown') + '. Open today ' + todayHours(store) + '.');
  lines.push('  Fulfilment: ' +
    (methods.length ? methods.join(' and ') : 'in store only') + '.');
  return lines.join('\n');
};

try {
  const token = await getToken();

  const listResponse = await fetch(API + '/store', {
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      Authorization: 'Bearer ' + token,
    },
  });
  if (!listResponse.ok) {
    throw new Error('GET /store returned HTTP ' + listResponse.status);
  }
  const stores = ((await listResponse.json()) || {}).stores || [];
  // Only stores flagged for ecommerce can hold a cart; the list also carries
  // click-and-collect-only and offline stores.
  const shoppable = stores.filter((store) => store.clickAndCollect ||
    store.delivery);

  const selectedId = currentStoreId();
  const selected = stores.find((store) => store.id === selectedId);

  const asked = (input || {}).store;
  if (!asked) {
    const lines = [];
    lines.push(selected
      ? 'Currently shopping at:\n' + describe(selected)
      : 'No New World store is selected yet.');
    lines.push('');
    lines.push(shoppable.length + ' stores can be shopped online. Ask for ' +
      'one by name to switch, for example "Greymouth".');
    return lines.join('\n');
  }

  const wanted = normalise(asked);
  const matches = shoppable.filter((store) =>
    normalise(store.name) === wanted);
  const candidates = matches.length ? matches : shoppable.filter((store) =>
    normalise(store.name).indexOf(wanted) !== -1);

  if (!candidates.length) {
    return 'No New World store matches "' + asked + '". Store names look ' +
      'like "New World Greymouth" or "New World Thorndon"; try a suburb or ' +
      'town.';
  }
  if (candidates.length > 1) {
    // A bare "new world" matches every store, so never print the whole list.
    const shown = candidates.slice(0, 15);
    const more = candidates.length - shown.length;
    return candidates.length + ' New World stores match "' + asked +
      '". Ask again with one of these exact names:\n' +
      shown.map((store) => '- ' + store.name).join('\n') +
      (more ? '\n...and ' + more + ' more. Narrow it down with a suburb or ' +
        'town name.' : '');
  }

  const store = candidates[0];
  if (store.id === selectedId) {
    return 'Already shopping at ' + store.name + '. Nothing changed.\n' +
      describe(store);
  }

  const changeResponse = await fetch(API + '/cart/store/' + store.id, {
    method: 'POST',
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    },
  });
  const changeBody = await changeResponse.json().catch(() => null);

  // 412 is the server refusing the move because the basket contains products
  // the new store does not sell. It names them in errorProducts.
  if (changeResponse.status === 412) {
    const blocked = ((changeBody || {}).errorProducts || [])
      .map((product) => '- ' + (product.name || product.productId) +
        ' (' + product.productId + ')');
    return 'Cannot switch to ' + store.name + ' while these items are in the ' +
      'cart, because that store does not sell them:\n' + blocked.join('\n') +
      '\nRemove them with update_cart (set quantity 0), then try again.';
  }
  if (!changeResponse.ok) {
    const reason = (changeBody || {}).errorReasonCode;
    if (reason === 'STORE_NOT_AVAILABLE_TO_CUSTOMER') {
      return store.name + ' is not available to this account for online ' +
        'shopping.';
    }
    return 'Could not switch to ' + store.name + ': ' +
      (reason || 'HTTP ' + changeResponse.status) + '.';
  }

  // The server has moved the cart; these cookies are what the page itself
  // reads on next render, so write them too or the UI keeps showing the old
  // store even though orders would go to the new one.
  writeCookie('eCom_STORE_ID', store.id);
  writeCookie('STORE_ID_V2', store.id + '|False');
  if (store.region) {
    writeCookie('Region', store.region);
  }

  return 'Now shopping at:\n' + describe(store) +
    '\n\nPrices and availability have been repriced against this store. ' +
    'Reload the page to see it reflected in the site UI.';
} catch (error) {
  return 'Could not change the New World store: ' +
    (error && error.message ? error.message : String(error));
}
