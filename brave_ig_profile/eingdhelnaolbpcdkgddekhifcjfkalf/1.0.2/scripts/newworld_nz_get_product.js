// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        get_product
// @match       https://www.newworld.co.nz/*
// @description Get the full detail of one New World product at the currently
// @description selected store: price, unit price, current special and when it
// @description ends, pack size, how it is sold, ingredients, allergens and
// @description nutrition. Defaults to the product on the page when called on a
// @description product page; otherwise pass a product id from search_catalog
// @description or browse_store.
// @schema      {"type":"object","properties":{"product":{"type":"string","description":"A product id such as \"5130351-EA-000\", or a newworld.co.nz product URL. Omit to describe the product shown on the current page."}}}
// ==/WebMCP==

// Product pages are client-rendered from GET /store/<storeId>/product/<id> on
// api-prod.newworld.co.nz, so read that directly rather than scraping. Price
// and availability are per store, so the store id is part of the path.
//
// Product ids look like "5130351-EA-000": an article number, the unit of
// measure (EA each, KGM per kilo) and a variant suffix. The site's own page
// URLs mangle that into "5130351_ea_000nw", so accept either spelling, plus a
// bare article number, and undo the mangling here.

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

const getStoreId = async () => {
  const id = readCookie('STORE_ID_V2').split('|')[0] ||
    readCookie('eCom_STORE_ID');
  if (id) {
    return id;
  }
  const response = await fetch('/next/api/stores/default', {
    credentials: 'include',
    headers: { Accept: 'application/json' },
  });
  const store = ((await response.json().catch(() => null)) || {}).data || {};
  return store.id;
};

// Turn anything the caller might hand us into candidate product ids. A bare
// article number is ambiguous between the each and per-kilo variants, so return
// both and let the lookup decide.
const candidateIds = (raw) => {
  let value = String(raw || '').trim();
  if (!value) {
    return [];
  }
  if (value.indexOf('/') !== -1) {
    value = value.split('?')[0].split('#')[0].replace(/\/+$/, '');
    value = value.split('/').pop();
  }
  value = value.split('?')[0];
  // Page URLs suffix the banner onto the slug: "5130351_ea_000nw".
  value = value.replace(/(nw|pns)$/i, '');
  value = value.split('_').join('-').toUpperCase();
  if (/^\d+$/.test(value)) {
    return [value + '-EA-000', value + '-KGM-000'];
  }
  return [value];
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

// GS1 unit-of-measure codes as they appear in the nutrition payload.
const UOMS = {
  GRM: 'g', MGM: 'mg', MC: 'ug', KJO: 'kJ', E14: 'kcal', MLT: 'ml',
};

const uom = (code) => (code ? (UOMS[code] || String(code).toLowerCase()) : '');

// The nutrients array interleaves two measurement bases for every nutrient:
// BY_MEASURE (the per-100g panel) and BY_SERVING. Reading it flat would print
// each nutrient twice with unrelated numbers, so fold the two bases into one
// row per nutrient, the way the site's own panel does. Restrict to the seven
// nutrients New World actually displays, in their order.
const CORE_NUTRIENTS = ['Energy', 'Protein', 'Fat - Total', 'Fat - Saturated',
  'Carbohydrates', 'Sugar - Total', 'Sodium'];

const amountText = (nutrient) => {
  const prefix = nutrient.measurementPrecision === 'LESS_THAN' ? '<'
    : nutrient.measurementPrecision === 'GREATER_THAN' ? '>' : '';
  const amount = nutrient.qtyContained;
  return prefix + (amount === undefined || amount === null ? '0' : amount) +
    uom(nutrient.nutrientUom);
};

const nutritionPanel = (info) => {
  const nutrients = (info || {}).nutrients || [];
  if (!nutrients.length) {
    return null;
  }
  const rows = {};
  nutrients.forEach((nutrient) => {
    const label = nutrient.nutrientTypeDescription;
    // PREPARED figures describe the made-up dish, not the packet as sold.
    if (!label || nutrient.preparationState === 'PREPARED' ||
        CORE_NUTRIENTS.indexOf(label) === -1) {
      return;
    }
    rows[label] = rows[label] || { label: label };
    if (nutrient.nutrientBasisQuantityType === 'BY_MEASURE') {
      rows[label].perMeasure = amountText(nutrient);
    } else if (nutrient.nutrientBasisQuantityType === 'BY_SERVING') {
      rows[label].perServing = amountText(nutrient);
    }
  });

  const ordered = CORE_NUTRIENTS
    .map((label) => rows[label])
    .filter((row) => row && (row.perMeasure || row.perServing));
  if (!ordered.length) {
    return null;
  }

  const basisOf = (type) => {
    const found = nutrients.find((nutrient) =>
      nutrient.nutrientBasisQuantityType === type);
    return found ? found.nutrientBasisQty + uom(found.nutrientBasisQtyUom) : '';
  };
  const measure = basisOf('BY_MEASURE');
  const serving = basisOf('BY_SERVING');
  const both = ordered.some((row) => row.perMeasure) &&
    ordered.some((row) => row.perServing);

  const heading = both
    ? 'Nutrition per ' + measure + ' / per ' + serving + ' serving'
    : 'Nutrition per ' + (measure || serving);
  const values = ordered.map((row) => row.label + ' ' +
    (both ? (row.perMeasure || '?') + ' / ' + (row.perServing || '?')
      : (row.perMeasure || row.perServing)));
  return heading + ': ' + values.join(', ') + '.';
};

try {
  const token = await getToken();
  const storeId = await getStoreId();
  if (!storeId) {
    return 'No New World store is selected, so prices cannot be looked up. ' +
      'Use change_store to pick one first.';
  }

  const asked = (input || {}).product ||
    (window.location.pathname.match(/\/shop\/product\/([^/?#]+)/) || [])[1];
  const ids = candidateIds(asked);
  if (!ids.length) {
    return 'No product was given, and this is not a New World product page. ' +
      'Pass a product id such as "5130351-EA-000", or use search_catalog to ' +
      'find one.';
  }

  let product = null;
  for (let i = 0; i < ids.length && !product; i += 1) {
    const response = await fetch(
      API + '/store/' + storeId + '/product/' + ids[i], {
        credentials: 'include',
        headers: {
          Accept: 'application/json',
          Authorization: 'Bearer ' + token,
        },
      });
    if (!response.ok) {
      continue;
    }
    const data = await response.json().catch(() => null);
    if (data && data.productId) {
      product = data;
    }
  }
  if (!product) {
    return 'No product ' + ids.join(' or ') + ' at the selected store. It ' +
      'may not be stocked there, or the id may be wrong; search_catalog ' +
      'returns ids that are valid for this store.';
  }

  const lines = [];
  const title = [product.brand, product.name].filter(Boolean).join(' ');
  lines.push(title + (product.displayName ? ', ' + product.displayName : ''));
  lines.push('Product id: ' + product.productId);

  const priceParts = [money(product.price)];
  if (product.comparativePricePerUnit) {
    priceParts.push(money(product.comparativePricePerUnit) + ' per ' +
      (product.comparativeUnitMeasureDescription ||
        product.comparativeUnitQuantityUoM));
  }
  lines.push('Price: ' + priceParts.join(', '));
  // nonLoyaltyCardPrice differs from price only when the shopper is seeing a
  // Clubcard rate, so mentioning it unconditionally would just be noise.
  if (product.nonLoyaltyCardPrice &&
      product.nonLoyaltyCardPrice !== product.price) {
    lines.push('Without Clubcard: ' + money(product.nonLoyaltyCardPrice));
  }

  // saleType BOTH means the item can be bought either as whole pieces or by
  // weight; the cart needs to be told which, so spell it out.
  if (product.saleType === 'WEIGHT') {
    lines.push('Sold by weight. Order in grams.');
  } else if (product.saleType === 'BOTH') {
    lines.push('Sold either by the piece or by weight.');
  }
  const weighable = product.weighable || {};
  if (weighable.minOrderQty || weighable.avgWeightPerUnit) {
    const notes = [];
    if (weighable.avgWeightPerUnit) {
      notes.push('about ' + weighable.avgWeightPerUnit +
        (weighable.avgWeightUoM || 'g') + ' each');
    }
    if (weighable.minOrderQty) {
      notes.push('minimum order ' + weighable.minOrderQty +
        (weighable.avgWeightUoM || 'g'));
    }
    if (weighable.stepSize) {
      notes.push('in steps of ' + weighable.stepSize +
        (weighable.avgWeightUoM || 'g'));
    }
    lines.push('Weight: ' + notes.join(', ') + '.');
  }

  (product.promotionList || []).forEach((promotion) => {
    if (promotion.suspended) {
      return;
    }
    const reward = (promotion.promotionRewards || [])[0] || {};
    const condition = (reward.promotionConditions || [])[0] || {};
    let text = 'Special: ' + (DECALS[promotion.decal] || 'Promotion') + ' ' +
      money(reward.rewardValue);
    if (Number(condition.thresholdQuantity) > 1) {
      text += ' each when you buy ' + condition.thresholdQuantity;
    }
    if (promotion.endDate) {
      text += ', until ' + promotion.endDate;
    }
    lines.push(text);
    if (promotion.ticketDescription) {
      lines.push('  Applies to: ' + promotion.ticketDescription);
    }
  });

  const tree = (product.categoryTrees || [])[0];
  if (tree) {
    lines.push('Category: ' +
      [tree.level0, tree.level1, tree.level2].filter(Boolean).join(' > '));
  }

  const availability = product.availability || [];
  if (availability.indexOf('ONLINE') === -1) {
    lines.push('Availability: in store only, cannot be added to an online ' +
      'order.');
  }
  const fulfilment = (product.fulfilmentOptions || [])
    .filter((option) => option.available)
    .map((option) => option.method.toLowerCase());
  if (fulfilment.length) {
    lines.push('Fulfilment: ' + fulfilment.join(', ') + '.');
  }
  if (product.restrictedFlag) {
    lines.push('Age restricted: requires ID on delivery or collection.');
  }
  if (product.originStatement) {
    lines.push('Origin: ' + product.originStatement);
  }

  const dietary = (product.facets || [])
    .map((facet) => facet.itemDescription)
    .filter(Boolean);
  if (dietary.length) {
    lines.push('Attributes: ' + dietary.join(', ') + '.');
  }

  if (product.description) {
    lines.push('');
    lines.push(String(product.description).replace(/\s+/g, ' ').trim());
  }
  if (product.ingredientStatement) {
    lines.push('');
    lines.push('Ingredients: ' +
      String(product.ingredientStatement).replace(/\s+/g, ' ').trim());
  }
  if (product.allergenStatement) {
    lines.push('Allergens: ' +
      String(product.allergenStatement).replace(/\s+/g, ' ').trim());
  }

  const nutrition = nutritionPanel(product.nutritionalInfo);
  if (nutrition) {
    lines.push('');
    lines.push(nutrition);
    const serves = (product.nutritionalInfo || {}).noServesPerPack;
    if (serves) {
      lines.push('Servings per pack: ' + serves + '.');
    }
  }

  lines.push('');
  lines.push('Add it with update_cart using id ' + product.productId + '.');
  return lines.join('\n');
} catch (error) {
  return 'Could not read the New World product: ' +
    (error && error.message ? error.message : String(error));
}
