// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        browse_store
// @match       https://www.newworld.co.nz/*
// @description Browse the aisles of the currently selected New World store.
// @description Call with no arguments to list the departments, then pass a
// @description category name to see its subcategories and the products in it.
// @description Use this when the shopper wants to explore what is available
// @description rather than look for something specific by name, which is what
// @description search_catalog is for.
// @schema      {"type":"object","properties":{"category":{"type":"string","description":"A category to open, for example \"Pantry\", \"Fruit & Vegetables\" or a full path like \"Pantry > Rice, Pasta & Grains\". Omit to list the top-level departments."},"page":{"type":"integer","minimum":1,"description":"1-based page number for the product list. Defaults to 1."},"size":{"type":"integer","minimum":1,"maximum":50,"description":"How many products to list, 1 to 50. Defaults to 20."}}}
// ==/WebMCP==

// The aisle navigation is a client-rendered mega-menu built from a per-store
// category tree, so it is only partly in the DOM and never for the categories
// the shopper has not hovered over. Fetch the tree instead:
// GET /store/<id>/categories returns three nested levels for that store.
//
// Listing the products in a category then goes through the same search endpoint
// search_catalog uses, filtered by an Algolia facet. The facet name encodes
// both the depth and the island - category0NI, category1SI and so on - so the
// depth a category was found at determines which facet to filter on, and the
// store's region determines the suffix.

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

const api = async (path, token, body) => {
  const response = await fetch(API + path, {
    method: body ? 'POST' : 'GET',
    credentials: 'include',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      Authorization: 'Bearer ' + token,
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(path + ' returned HTTP ' + response.status +
      ((data && data.message) ? ' (' + data.message + ')' : ''));
  }
  return data;
};

// The island matters twice over here: category facets are named category0NI or
// category0SI, so getting it wrong returns an empty aisle rather than a wrong
// order. The "Region" cookie is only written when the shopper actively changes
// store, so fall back to the store record itself rather than to the server's
// default store, which may be on the other island.
const getStore = async (token) => {
  let id = readCookie('STORE_ID_V2').split('|')[0] ||
    readCookie('eCom_STORE_ID');
  const region = readCookie('Region');
  if (id && region) {
    return { id: id, region: region };
  }
  if (!id) {
    const response = await fetch('/next/api/stores/default', {
      credentials: 'include',
      headers: { Accept: 'application/json' },
    });
    const store = ((await response.json().catch(() => null)) || {}).data || {};
    id = store.id;
    if (id && store.region) {
      return { id: id, region: store.region };
    }
  }
  if (!id) {
    return { id: '', region: 'NI' };
  }
  const detail = await api('/store/' + id, token).catch(() => null);
  return { id: id, region: (detail && detail.region) || 'NI' };
};

const money = (cents) => {
  if (cents === undefined || cents === null) {
    return 'unknown';
  }
  return '$' + (Number(cents) / 100).toFixed(2);
};

const DECALS = {
  1000: 'Saver',
  2000: 'Store Saver',
  3000: 'Super Saver',
  4000: 'Club Deal',
  4700: 'Everyday Low Price',
  4701: 'Everyday Low Price',
  5000: 'Super Saver Club Deal',
};

const promotionLabel = (product) => {
  const promotions = product.promotions || [];
  const promotion = promotions.find((entry) => entry.bestPromotion) ||
    promotions[0];
  if (!promotion) {
    return '';
  }
  let label = ', ' + (DECALS[promotion.decal] || 'Special') + ' ' +
    money(promotion.rewardValue);
  if (Number(promotion.threshold) > 1) {
    label += ' each when you buy ' + promotion.threshold;
  }
  if (promotion.cardDependencyFlag) {
    label += ' with Clubcard';
  }
  return label;
};

const normalise = (value) =>
  String(value === undefined || value === null ? '' : value)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

// Flatten the tree into (node, depth, path) so a category can be found by its
// own name or by a "Parent > Child" path without walking the tree per query.
const flatten = (nodes, depth, trail, out) => {
  (nodes || []).forEach((node) => {
    if (!node || !node.name) {
      return;
    }
    const path = trail.concat([node.name]);
    out.push({ node: node, depth: depth, path: path });
    flatten(node.children, depth + 1, path, out);
  });
  return out;
};

try {
  const token = await getToken();
  const store = await getStore(token);
  if (!store.id) {
    return 'No New World store is selected, so there are no aisles to ' +
      'browse. Use change_store to pick one first.';
  }

  const tree = await api('/store/' + store.id + '/categories', token);
  const departments = (tree || []).filter((node) => node && node.name);
  const asked = String((input || {}).category || '').trim();

  if (!asked) {
    const lines = [];
    lines.push('Departments in the selected New World store:');
    departments.forEach((department) => {
      const children = (department.children || [])
        .map((child) => child.name)
        .filter(Boolean);
      lines.push('- ' + department.name + (children.length
        ? ' (' + children.length + ' subcategor' +
          (children.length === 1 ? 'y' : 'ies') + ')'
        : ''));
    });
    lines.push('');
    lines.push('Pass one of these as "category" to see what is in it.');
    return lines.join('\n');
  }

  const all = flatten(departments, 0, [], []);
  const wantedPath = normalise(asked.split('>').join(''));
  const wantedLeaf = normalise(asked.split('>').pop());

  // Prefer a full-path match so "Pantry > Rice" beats a bare "Rice" elsewhere
  // in the tree, then an exact leaf name, then a substring.
  let found = all.filter((entry) =>
    normalise(entry.path.join('')) === wantedPath);
  if (!found.length) {
    found = all.filter((entry) =>
      normalise(entry.node.name) === wantedLeaf);
  }
  if (!found.length) {
    found = all.filter((entry) =>
      normalise(entry.node.name).indexOf(wantedLeaf) !== -1);
  }

  if (!found.length) {
    // A path like "Pantry > Nope" is a near miss worth answering usefully, so
    // if the parent part of the path does resolve, list what it actually holds.
    const parentName = normalise(asked.split('>').slice(0, -1).join(''));
    const parent = parentName && all.find((entry) =>
      normalise(entry.path.join('')) === parentName);
    if (parent) {
      return 'No category "' + asked.split('>').pop().trim() + '" under ' +
        parent.path.join(' > ') + '. It contains: ' +
        (parent.node.children || []).map((child) => child.name)
          .filter(Boolean).join(', ') + '.';
    }
    return 'No category matches "' + asked + '". The departments are: ' +
      departments.map((department) => department.name).join(', ') + '.';
  }
  if (found.length > 1) {
    const shown = found.slice(0, 15);
    return found.length + ' categories match "' + asked +
      '". Ask again with one of these full paths:\n' +
      shown.map((entry) => '- ' + entry.path.join(' > ')).join('\n') +
      (found.length > shown.length
        ? '\n...and ' + (found.length - shown.length) + ' more.' : '');
  }

  const target = found[0];
  const children = (target.node.children || [])
    .map((child) => child.name)
    .filter(Boolean);

  const args = input || {};
  const size = Math.min(Math.max(parseInt(args.size, 10) || 20, 1), 50);
  const page = Math.max(parseInt(args.page, 10) || 1, 1) - 1;
  const facet = 'category' + target.depth + store.region + ':' +
    target.node.name;

  const result = await api('/search/paginated/products', token, {
    algoliaQuery: {
      query: '',
      facetFilters: [[facet], ['tobacco:false']],
      hitsPerPage: size,
      page: page,
    },
    storeId: store.id,
    hitsPerPage: size,
    page: page,
    sortOrder: store.region === 'SI' ? 'SI_POPULARITY_ASC'
      : 'NI_POPULARITY_ASC',
    // Browsing should show the aisle, not New World's sponsored placements,
    // and should not report ad impressions on the shopper's behalf.
    precisionMedia: {
      adDomain: 'CATEGORY_PAGE',
      adPositions: [],
      publishImpressionEvent: false,
      disableAds: true,
    },
  });

  const lines = [];
  lines.push(target.path.join(' > '));
  if (children.length) {
    lines.push('');
    lines.push('Subcategories: ' + children.join(', ') + '.');
  }

  const products = result.products || [];
  lines.push('');
  if (!products.length) {
    lines.push('No products are listed in this category at the selected ' +
      'store.');
    return lines.join('\n');
  }
  lines.push(result.totalHits + ' products in this category, most popular ' +
    'first. Showing page ' + (page + 1) + ' of ' + result.totalPages + '.');
  lines.push('');
  products.forEach((product, index) => {
    const title = [product.brand, product.name].filter(Boolean).join(' ');
    const price = product.singlePrice || {};
    const unit = price.comparativePrice;
    lines.push((index + 1) + '. ' + title +
      (product.displayName ? ', ' + product.displayName : '') + ' - ' +
      money(price.price) + promotionLabel(product) +
      (unit && unit.pricePerUnit
        ? ' (' + money(unit.pricePerUnit) + ' per ' +
          (unit.measureDescription || unit.unitQuantityUom) + ')' : ''));
    lines.push('   id ' + product.productId +
      (product.saleType === 'WEIGHT' ? ' - sold by weight' : ''));
  });
  lines.push('');
  lines.push('Use the id with get_product for detail, or with update_cart to ' +
    'add it to the basket.');
  return lines.join('\n');
} catch (error) {
  return 'Could not browse the New World store: ' +
    (error && error.message ? error.message : String(error));
}
