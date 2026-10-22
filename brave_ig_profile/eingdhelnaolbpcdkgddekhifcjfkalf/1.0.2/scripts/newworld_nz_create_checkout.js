// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        create_checkout
// @match       https://www.newworld.co.nz/*
// @description Get the New World cart ready to check out and hand back the
// @description checkout link. Reprices the basket, reports the subtotal, fees
// @description and estimated total, and flags anything that would block the
// @description order such as unavailable items or not being signed in. It does
// @description not place the order or pay for it: the shopper finishes at the
// @description returned URL, where they pick a delivery or collection slot and
// @description confirm payment.
// @schema      {"type":"object","properties":{}}
// ==/WebMCP==

// This stops deliberately short of ordering. The same API can place a real,
// charged grocery order (POST /order, then POST /order/pay against a stored
// card), and neither is reversible from here, so a tool call should not be able
// to reach them. What it does instead is everything up to that point:
//
//   POST /cart/validate is what the site runs before letting the shopper into
//   checkout. It reprices against current store prices and either returns the
//   fresh cart or an errorReasonCode naming the blocker. Surfacing that here
//   means the shopper hits the checkout page with a basket that will actually
//   go through, rather than discovering the problem after picking a timeslot.
//
// Delivery and collection fees are not included in the estimate: they depend on
// the timeslot and the fulfilment method, which are chosen on the checkout page
// itself and so are not knowable yet.

const API = 'https://api-prod.newworld.co.nz/v1/edge';
const CHECKOUT_URL = 'https://www.newworld.co.nz/shop/shopping-cart';

// The site's own wording for the blockers /cart/validate can report.
const BLOCKERS = {
  ORDER_HAS_UNAVAILABLE_ITEMS: 'the cart contains items this store cannot ' +
    'supply',
  ORDER_HAS_TOO_MANY_ITEMS_FOR_QUICK_COMM_TIMESLOT: 'the cart has too many ' +
    'items for the chosen express timeslot',
  NO_QUICK_COMM_SPACES_AVAILABLE: 'express delivery is currently unavailable',
  QUICK_COMM_ORDER_REQUIRES_MOBILE_NUMBER: 'express delivery needs a mobile ' +
    'number on the account',
  INVALID_PROMO_CODE_PRIORITY_TYPE: 'the promo code is not valid for the ' +
    'chosen timeslot',
  INVALID_PROMO_CODE_REGION: 'the promo code is not valid in this region',
  INVALID_PROMO_CODE_STORE: 'the promo code is not valid at this store',
  INVALID_PROMO_CODE_SERVICE_TYPE: 'the promo code is not valid for this ' +
    'service type',
  AUTHENTICATION_REQUIRED: 'the shopper needs to sign in',
};

// This route exchanges the HttpOnly session cookies for a short-lived bearer
// token, and provisions an anonymous session when there is no session at all,
// so it covers a first-time visitor on its own. There is deliberately no
// fallback to /next/api/user/login/guest: that mints a *new* session and
// rewrites refresh_token, so reaching for it when this call fails transiently
// would sign a logged-in shopper out and strand the cart they had going.
const getSession = async () => {
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

// The token is a plain JWT whose payload carries the account's roles, so who is
// shopping can be reported without a second request. Signature is irrelevant
// here: this is only ever used to tell the shopper what they are already
// entitled to see about themselves.
const identity = (token) => {
  try {
    const payload = String(token).split('.')[1];
    const json = atob(payload.split('-').join('+').split('_').join('/'));
    const claims = JSON.parse(json);
    const roles = claims.roles || [];
    return {
      guest: roles.indexOf('ANONYMOUS') !== -1,
      name: claims.firstName && claims.firstName !== 'anonymous'
        ? claims.firstName : null,
    };
  } catch (error) {
    return { guest: false, name: null };
  }
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
  const token = await getSession();
  const who = identity(token);
  const headers = {
    Accept: 'application/json',
    'Content-Type': 'application/json',
    Authorization: 'Bearer ' + token,
  };

  const cartResponse = await fetch(API + '/cart', {
    credentials: 'include',
    headers: headers,
  });
  if (!cartResponse.ok) {
    throw new Error('GET /cart returned HTTP ' + cartResponse.status);
  }
  const current = (await cartResponse.json()) || {};
  if (!(current.products || []).length) {
    return 'The New World cart is empty, so there is nothing to check out. ' +
      'Add items with update_cart first.';
  }

  const validateResponse = await fetch(API + '/cart/validate', {
    method: 'POST',
    credentials: 'include',
    headers: headers,
  });
  const validated = await validateResponse.json().catch(() => null);

  const lines = [];

  if (!validateResponse.ok || (validated && validated.errorReasonCode)) {
    const code = (validated || {}).errorReasonCode;
    const blocker = BLOCKERS[code];
    lines.push('The cart is not ready to check out: ' +
      (blocker || 'the site reported ' + (code || 'HTTP ' +
        validateResponse.status)) + '.');
    const offenders = (validated || {}).errorProducts || [];
    if (offenders.length) {
      lines.push('');
      lines.push('Problem items, removable with update_cart at quantity 0:');
      offenders.forEach((line) => {
        lines.push('- ' + [line.brand, line.name].filter(Boolean).join(' ') +
          ' (' + line.productId + ')');
      });
    }
    return lines.join('\n');
  }

  // On success validate answers with the freshly repriced cart, so prefer it
  // over the copy read a moment ago; prices can move between the two calls.
  const cart = validated && validated.products ? validated : current;
  const products = cart.products || [];
  const store = cart.store || {};

  lines.push('Ready to check out at ' + (store.storeName || 'the selected ' +
    'store') + '.');
  lines.push('');
  products.forEach((line) => {
    lines.push('- ' + [line.brand, line.name].filter(Boolean).join(' ') + ' ' +
      quantityText(line) + ' - ' + money(line.price));
  });

  const unavailable = cart.unavailableProducts || [];
  if (unavailable.length) {
    lines.push('');
    lines.push('Excluded because this store cannot supply them:');
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
  const fees = (Number(cart.serviceFee) || 0) + (Number(cart.bagFee) || 0);
  const discounts = (Number(cart.promoCodeDiscount) || 0) +
    (Number(cart.subscriptionDiscount) || 0);
  lines.push('Estimated total: ' +
    money(fees + (Number(cart.subtotal) || 0) + discounts) +
    ', before any delivery or collection fee.');

  lines.push('');
  const todo = [];
  if (who.guest) {
    todo.push('Sign in. The basket is currently held as a guest, so Club ' +
      'Deal prices and saved payment cards are not applied.');
  }
  todo.push('Choose a delivery or collection timeslot, which sets the ' +
    'fulfilment fee.');
  todo.push('Confirm payment.');
  lines.push('Still to do, in the browser:');
  todo.forEach((step, index) => {
    lines.push((index + 1) + '. ' + step);
  });

  lines.push('');
  lines.push('Finish the order at ' + CHECKOUT_URL);
  lines.push('This tool has not placed or paid for the order, and will not; ' +
    'that has to be confirmed by the shopper on that page.');
  return lines.join('\n');
} catch (error) {
  return 'Could not prepare the New World checkout: ' +
    (error && error.message ? error.message : String(error));
}
