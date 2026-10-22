// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        search_catalog
// @match       https://www.newworld.co.nz/*
// @description Search the New World grocery catalogue for the currently
// @description selected store and return matching products with their price,
// @description unit price, any current special, and the product id needed to
// @description add them to the cart. Results are store-specific: use
// @description change_store first if the shopper wants a different store.
// @schema      {"type":"object","properties":{"query":{"type":"string","description":"What to search for, for example \"oat milk\", \"free range eggs\" or \"whittakers dark chocolate\"."},"sort":{"type":"string","enum":["popularity","price_asc","price_desc","unit_price_asc","unit_price_desc"],"description":"Result ordering. Defaults to popularity. Use unit_price_asc to find the best value per kg or per litre."},"page":{"type":"integer","minimum":1,"description":"1-based page number, for paging through more results. Defaults to 1."},"size":{"type":"integer","minimum":1,"maximum":50,"description":"How many products to return, 1 to 50. Defaults to 20."},"specials_only":{"type":"boolean","description":"Only return products that currently have a promotion. Defaults to false."}}}
// ==/WebMCP==

// Search results are client-rendered from an Algolia-backed endpoint, so there
// is nothing useful to scrape until the shopper has already searched. Instead
// call the same API the page uses: POST /search/paginated/products on
// api-prod.newworld.co.nz/v1/edge.
//
// Two deliberate choices in the request body:
//   - precisionMedia.disableAds, so the tool returns the actual catalogue
//     rather than New World's sponsored placements interleaved at positions
//     4/8/12 the way the website shows them.
//   - publishImpressionEvent false, so reading the catalogue does not emit ad
//     impression telemetry on the shopper's behalf.
//
// The endpoint rejects a request with no sortOrder, and the popularity index is
// per island, so the store's region has to be resolved before searching.

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

// The selected store lives in a readable cookie as "<uuid>|<isInitialStore>".
// Its island has to come with it, because the popularity index is per island.
// The "Region" cookie only gets written when the shopper actively changes
// store, so when it is missing resolve the island from the store record itself
// rather than from the server's default store, which may well be on the other
// island and would silently sort results by the wrong index.
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

// Promotion decals map to New World's marketing names; the shopper sees these
// words on the shelf tickets, so use them rather than the raw code.
const DECALS = {
  1000: 'Saver',
  2000: 'Store Saver',
  3000: 'Super Saver',
  4000: 'Club Deal',
  4700: 'Everyday Low Price',
  4701: 'Everyday Low Price',
  5000: 'Super Saver Club Deal',
};

const bestPromotion = (product) => {
  const promotions = product.promotions || [];
  if (!promotions.length) {
    return null;
  }
  return promotions.find((promotion) => promotion.bestPromotion) ||
    promotions[0];
};

const promotionLabel = (promotion) => {
  const name = DECALS[promotion.decal] || 'Special';
  let label = name + ' ' + money(promotion.rewardValue);
  if (Number(promotion.threshold) > 1) {
    label += ' each when you buy ' + promotion.threshold;
  }
  if (promotion.cardDependencyFlag) {
    label += ' with Clubcard';
  }
  return label;
};

const comparative = (price) => {
  const unit = (price || {}).comparativePrice;
  if (!unit || !unit.pricePerUnit) {
    return '';
  }
  return ' (' + money(unit.pricePerUnit) + ' per ' +
    (unit.measureDescription || unit.unitQuantityUom) + ')';
};

const describe = (product, index) => {
  const title = [product.brand, product.name].filter(Boolean).join(' ');
  const size = product.displayName ? ', ' + product.displayName : '';
  const price = product.singlePrice || {};
  const parts = [];
  parts.push(money(price.price));
  const promotion = bestPromotion(product);
  if (promotion) {
    parts.push(promotionLabel(promotion));
  }
  const flags = [];
  if (product.saleType === 'WEIGHT') {
    flags.push('sold by weight');
  }
  if (product.restrictedFlag) {
    flags.push('age restricted');
  }
  if ((product.availability || []).indexOf('ONLINE') === -1) {
    flags.push('in store only, cannot be ordered online');
  }
  const lines = [];
  lines.push((index + 1) + '. ' + title + size + ' - ' + parts.join(', ') +
    comparative(price));
  lines.push('   id ' + product.productId +
    (flags.length ? ' - ' + flags.join(', ') : ''));
  return lines.join('\n');
};

const SORTS = {
  price_asc: 'PRICE_ASC',
  price_desc: 'PRICE_DESC',
  unit_price_asc: 'UNIT_PRICE_ASC',
  unit_price_desc: 'UNIT_PRICE_DESC',
};

const args = input || {};
const query = String(args.query || '').trim();
if (!query) {
  return 'No search term was given. Say what to look for, for example "oat ' +
    'milk" or "free range eggs".';
}

try {
  const token = await getToken();
  const store = await getStore(token);
  if (!store.id) {
    return 'No New World store is selected, so there is no catalogue to ' +
      'search. Use change_store to pick one first.';
  }

  const size = Math.min(Math.max(parseInt(args.size, 10) || 20, 1), 50);
  // The API pages from zero; the tool takes a 1-based page for the caller.
  const page = Math.max(parseInt(args.page, 10) || 1, 1) - 1;
  const sortOrder = SORTS[args.sort] ||
    (store.region === 'SI' ? 'SI_POPULARITY_ASC' : 'NI_POPULARITY_ASC');

  const result = await api('/search/paginated/products', token, {
    algoliaQuery: {
      query: query,
      // Tobacco is excluded from the storefront catalogue by default and is
      // fetched separately by the site; keep it out here too.
      facetFilters: [['tobacco:false']],
      hitsPerPage: size,
      page: page,
    },
    storeId: store.id,
    hitsPerPage: size,
    page: page,
    sortOrder: sortOrder,
    precisionMedia: {
      adDomain: 'SEARCH_PAGE',
      adPositions: [],
      publishImpressionEvent: false,
      disableAds: true,
    },
  });

  const matched = result.products || [];
  // The API has no "on special" facet, so this filter can only apply to the
  // page already fetched. Say so, or the counts look inconsistent.
  const products = args.specials_only
    ? matched.filter((product) => bestPromotion(product))
    : matched;
  if (!products.length) {
    return 'No products matched "' + query + '" at this store' +
      (args.specials_only ? ' with a current special. ' + matched.length +
        ' results on this page were all full price; try another page.' : '.');
  }

  const lines = [];
  lines.push('Found ' + result.totalHits + ' products matching "' + query +
    '" at the selected store. Showing page ' + (page + 1) + ' of ' +
    result.totalPages + (args.specials_only
      ? ', narrowed to the ' + products.length + ' of ' + matched.length +
        ' results on this page that are on special'
      : '') + '. Prices are NZD and store-specific.');
  lines.push('');
  products.forEach((product, index) => {
    lines.push(describe(product, index));
  });
  lines.push('');
  lines.push('Use the id with get_product for detail, or with update_cart to ' +
    'add it to the basket.');
  return lines.join('\n');
} catch (error) {
  return 'Could not search the New World catalogue: ' +
    (error && error.message ? error.message : String(error));
}
