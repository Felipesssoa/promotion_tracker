// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        update_cart
// @match       https://www.newworld.co.nz/*
// @description Add products to the New World cart, change how much of
// @description something is in it, or take something out. Quantities are set
// @description outright, not added to what is already there, so read get_cart
// @description first if you mean "two more"; a quantity of 0 removes the line.
// @description Items sold by weight are ordered in grams instead. Several
// @description items can be changed in one call.
// @schema      {"type":"object","properties":{"items":{"type":"array","minItems":1,"description":"The lines to set. Each needs a product id and either a quantity or, for items sold by weight, grams.","items":{"type":"object","properties":{"product":{"type":"string","description":"Product id from search_catalog, browse_store or get_product, for example \"5130351-EA-000\"."},"quantity":{"type":"integer","minimum":0,"description":"How many of this item the cart should end up with. 0 removes it. Use for items sold by the piece."},"grams":{"type":"integer","minimum":0,"description":"How many grams the cart should end up with. 0 removes it. Use for items sold by weight, such as loose fruit or deli meat."}},"required":["product"]}}},"required":["items"]}
// ==/WebMCP==

// Adding to the cart on this site is POST /cart on api-prod.newworld.co.nz with
// a products array. Two things about that endpoint drive the design here:
//
//   - Quantities are absolute, not deltas. The site's own "+" button reads the
//     current line and posts the new total. This tool does the same, and says
//     so in its description, because an LLM that assumed "add" would silently
//     double orders.
//   - "quantity" means pieces for a UNITS product but *grams* for a WEIGHT one,
//     and must be a whole number either way. A product's saleType can also be
//     BOTH, meaning it can be bought either way, so which one is intended has
//     to be resolved per line before posting.
//
// Each line is therefore looked up first via GET /store/<storeId>/product/<id>,
// which validates the id against this store, gives the sale type, and supplies
// the minimum order weight so a doomed request can be refused with a useful
// message instead of a generic "Unable to price products" from the server.

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

const selectedStoreId = () =>
  readCookie('STORE_ID_V2').split('|')[0] || readCookie('eCom_STORE_ID');

const call = async (path, token, method, body) => {
  const response = await fetch(API + path, {
    method: method,
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  return {
    ok: response.ok,
    status: response.status,
    data: await response.json().catch(() => null),
  };
};

const money = (cents) => '$' + (Number(cents || 0) / 100).toFixed(2);

const plural = (count, noun) => count + ' ' + noun + (count === 1 ? '' : 's');

// WEIGHT lines are ordered in whole grams. The Number round-trip drops the
// trailing zeros toFixed adds without touching significant ones, so 10000g
// reads as 10kg rather than 1kg.
const quantityText = (saleType, quantity) => {
  if (saleType !== 'WEIGHT') {
    return 'x' + quantity;
  }
  const grams = Number(quantity) || 0;
  return grams >= 1000
    ? String(Number((grams / 1000).toFixed(3))) + 'kg' : grams + 'g';
};

// Accept a bare id, a page-style slug ("5130351_ea_000nw") or a full URL.
const normaliseId = (raw) => {
  let value = String(raw || '').trim();
  if (!value) {
    return '';
  }
  if (value.indexOf('/') !== -1) {
    value = value.split('?')[0].split('#')[0].replace(/\/+$/, '').split('/')
      .pop();
  }
  return value.split('?')[0].replace(/(nw|pns)$/i, '').split('_').join('-')
    .toUpperCase();
};

const args = input || {};
const items = Array.isArray(args.items) ? args.items : [];
if (!items.length) {
  return 'No items were given. Pass items as [{"product":"5130351-EA-000",' +
    '"quantity":2}], using a quantity of 0 to remove something.';
}

try {
  const token = await getToken();
  const storeId = selectedStoreId();
  if (!storeId) {
    return 'No New World store is selected, so there is nothing to add to. ' +
      'Use change_store to pick one first.';
  }

  const before = (await call('/cart', token, 'GET')).data || {};
  const beforeLines = {};
  (before.products || []).forEach((line) => {
    beforeLines[line.productId] = line;
  });

  // POST /cart answers "Store is not defined" until the cart has been bound to
  // a store. Binding an empty cart is harmless, so do it rather than failing.
  if (!(before.store || {}).storeId) {
    await call('/cart/store/' + storeId, token, 'POST');
  }

  const wanted = [];
  const problems = [];
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i] || {};
    const id = normaliseId(item.product);
    if (!id) {
      problems.push('An item was given with no product id.');
      continue;
    }

    const lookup = await call('/store/' + storeId + '/product/' + id, token,
      'GET');
    const product = lookup.ok ? lookup.data : null;
    if (!product || !product.productId) {
      problems.push(id + ' is not stocked at this store, or is not a valid ' +
        'product id.');
      continue;
    }

    const byWeight = item.grams !== undefined && item.grams !== null;
    const raw = byWeight ? item.grams : item.quantity;
    if (raw === undefined || raw === null) {
      problems.push(id + ' was given without a quantity or grams.');
      continue;
    }
    const quantity = Number(raw);
    if (!Number.isFinite(quantity) || quantity < 0 ||
        Math.floor(quantity) !== quantity) {
      problems.push(id + ' needs a whole number; got ' + raw + '.');
      continue;
    }

    // saleType is UNITS, WEIGHT, or BOTH when the item can be bought either
    // way. Only BOTH gives the caller a real choice.
    const saleType = product.saleType;
    let resolved;
    if (byWeight) {
      if (saleType === 'UNITS') {
        problems.push([product.brand, product.name].filter(Boolean).join(' ') +
          ' (' + id + ') is sold by the piece, not by weight. Use quantity ' +
          'instead of grams.');
        continue;
      }
      resolved = 'WEIGHT';
    } else {
      if (saleType === 'WEIGHT') {
        problems.push([product.brand, product.name].filter(Boolean).join(' ') +
          ' (' + id + ') is sold by weight. Use grams instead of quantity.');
        continue;
      }
      resolved = 'UNITS';
    }

    const weighable = product.weighable || {};
    if (resolved === 'WEIGHT' && quantity > 0 && weighable.minOrderQty &&
        quantity < weighable.minOrderQty) {
      problems.push([product.brand, product.name].filter(Boolean).join(' ') +
        ' (' + id + ') has a minimum order of ' + weighable.minOrderQty +
        'g; ' + quantity + 'g was asked for.');
      continue;
    }

    wanted.push({
      productId: product.productId,
      quantity: quantity,
      sale_type: resolved,
      name: [product.brand, product.name].filter(Boolean).join(' '),
    });
  }

  if (!wanted.length) {
    return 'Nothing was changed.\n' + problems.map((p) => '- ' + p).join('\n');
  }

  const result = await call('/cart', token, 'POST', {
    products: wanted.map((line) => ({
      productId: line.productId,
      quantity: line.quantity,
      sale_type: line.sale_type,
    })),
  });

  if (!result.ok) {
    const data = result.data || {};
    // The server reports pricing failures per product rather than as a message.
    const failed = (data.products || [])
      .map((line) => line.productId)
      .join(', ');
    return 'The cart was not changed: ' +
      (data.message || 'HTTP ' + result.status) +
      (failed ? ' (' + failed + ')' : '') + '.' +
      (problems.length ? '\n' + problems.map((p) => '- ' + p).join('\n') : '');
  }

  const after = result.data || {};
  const afterLines = {};
  (after.products || []).forEach((line) => {
    afterLines[line.productId] = line;
  });

  const changes = wanted.map((line) => {
    const was = beforeLines[line.productId];
    const now = afterLines[line.productId];
    if (!now) {
      return was ? 'Removed ' + line.name
        : line.name + ' was already not in the cart.';
    }
    const nowText = quantityText(now.sale_type, now.quantity) + ' - ' +
      money(now.price);
    if (!was) {
      return 'Added ' + line.name + ' ' + nowText;
    }
    if (was.quantity === now.quantity && was.sale_type === now.sale_type) {
      return line.name + ' unchanged at ' + nowText;
    }
    return 'Changed ' + line.name + ' from ' +
      quantityText(was.sale_type, was.quantity) + ' to ' + nowText;
  });

  const lines = [];
  lines.push('Cart at ' + ((after.store || {}).storeName || 'the selected ' +
    'store') + ':');
  changes.forEach((change) => {
    lines.push('- ' + change);
  });
  if (problems.length) {
    lines.push('');
    lines.push('Not changed:');
    problems.forEach((problem) => {
      lines.push('- ' + problem);
    });
  }

  const products = after.products || [];
  const itemCount = products.reduce((total, line) => total +
    (line.sale_type === 'WEIGHT' ? 1 : (Number(line.quantity) || 0)), 0);
  const fees = (Number(after.serviceFee) || 0) + (Number(after.bagFee) || 0);
  const discounts = (Number(after.promoCodeDiscount) || 0) +
    (Number(after.subscriptionDiscount) || 0);
  lines.push('');
  lines.push('Cart now holds ' + plural(products.length, 'line') + ', ' +
    plural(itemCount, 'item') + '. Subtotal ' + money(after.subtotal) +
    ', estimated total ' +
    money(fees + (Number(after.subtotal) || 0) + discounts) +
    ' including fees.');

  const unavailable = after.unavailableProducts || [];
  if (unavailable.length) {
    lines.push('Note: ' + unavailable.length + ' item(s) already in the cart ' +
      'are unavailable at this store and are excluded from the total.');
  }
  return lines.join('\n');
} catch (error) {
  return 'Could not update the New World cart: ' +
    (error && error.message ? error.message : String(error));
}
