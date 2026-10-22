// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        region_forecast
// @match       https://www.avalanche.net.nz/*
// @description Report the current New Zealand Avalanche Advisory forecast for
// @description one region: the avalanche danger rating for each elevation
// @description band, the forecast avalanche problems, the forecaster's
// @description confidence, and the recent activity, snowpack and mountain
// @description weather summaries. Defaults to the region shown on the current
// @description page; pass "region" to ask about a different one.
// @schema      {"type":"object","properties":{"region":{"type":"string","description":"NZAA region name or URL slug, for example \"Wanaka\", \"Aoraki/Mt Cook\" or \"nelson-lakes\". Omit to use the region shown on the current page."}}}
// ==/WebMCP==

// avalanche.net.nz is a Vue single-page app. The forecast is client-rendered
// and the region picker is a Mapbox canvas, so there is little dependable
// forecast text in the DOM. Read the site's own unauthenticated, same-origin
// JSON API instead: /api/region lists the forecast regions and /api/forecast
// returns the two most recent forecasts for every region (newest first).
// /api/forecast/<id> ignores the id and returns the same full payload, so the
// filtering has to happen here. The page shows the newest forecast for its
// region, so pick the newest "created" timestamp to stay consistent with what
// the user is looking at.

const DANGER_RATINGS = {
  '-5': 'Spring Conditions',
  '-4': 'No Specific Avalanche Type',
  '-3': 'Centre Closed',
  '-2': 'Insufficient Snow',
  '-1': 'Insufficient Information',
  '0': 'No Rating',
  '1': 'Low',
  '2': 'Moderate',
  '3': 'Considerable',
  '4': 'High',
  '5': 'Extreme',
};

// The aspect rose is keyed by elevation band: high alpine, alpine, sub-alpine.
const ROSE_BANDS = [
  ['ha', 'high alpine'],
  ['a', 'alpine'],
  ['sa', 'sub-alpine'],
];

const ASPECTS = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'];

const TRENDS = { NoChange: 'no change' };

const getJson = async (path) => {
  const response = await fetch(path, {
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw new Error('GET ' + path + ' returned HTTP ' + response.status);
  }
  return response.json();
};

// Forecast prose is authored in a CMS and stored as HTML. Parse it in an inert
// document so nothing is executed or fetched, then collapse whitespace: the
// editor leaves long runs of &nbsp; behind, which \s matches as U+00A0.
const toText = (html) => {
  if (!html) {
    return '';
  }
  const doc = new DOMParser().parseFromString(String(html), 'text/html');
  return doc.body.textContent.replace(/\s+/g, ' ').trim();
};

const normalise = (value) =>
  String(value === undefined || value === null ? '' : value)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

// Each altitudeDanger entry carries only its boundaries; the band it describes
// comes from its position in the array, the same way the site's
// AltitudeDangerRow component labels the three rows.
const bandLabel = (band, index) => {
  if (index === 0) {
    return 'Above ' + band.altitudeFrom + ' metres';
  }
  if (index === 1) {
    return band.altitudeTo + ' - ' + band.altitudeFrom + ' metres';
  }
  return 'Below ' + band.altitudeFrom + ' metres';
};

const ratingLabel = (rating) => {
  const label = DANGER_RATINGS[String(rating)];
  return (label || 'Unknown') + ' (' + rating + ')';
};

// API timestamps are NZ wall-clock strings with no offset, and a forecast is
// valid for validPeriod hours from "created". Do the arithmetic in UTC so the
// viewer's own time zone cannot shift the answer.
const validUntil = (created, validPeriod) => {
  const parts = String(created)
    .match(/^(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d)/);
  if (!parts) {
    return 'unknown';
  }
  const pad = (value) => String(value).padStart(2, '0');
  const hours = parseInt(validPeriod, 10) || 24;
  const end = new Date(Date.UTC(Number(parts[1]), Number(parts[2]) - 1,
    Number(parts[3]), Number(parts[4]), Number(parts[5])) + hours * 3600000);
  return end.getUTCFullYear() + '-' + pad(end.getUTCMonth() + 1) + '-' +
    pad(end.getUTCDate()) + ' ' + pad(end.getUTCHours()) + ':' +
    pad(end.getUTCMinutes());
};

// Problem time windows arrive as HH:MM:SS; the seconds are always zero.
const hourMinute = (time) => String(time || '').slice(0, 5) || 'unknown';

// Forecasters often leave the rose blank, so only report bands that actually
// have an aspect flagged rather than implying "no aspects are affected".
const roseLines = (aspects) => {
  const lines = [];
  ROSE_BANDS.forEach((band) => {
    const rose = (aspects || {})[band[0]];
    if (!rose) {
      return;
    }
    const flagged = ASPECTS
      .filter((aspect) => Number(rose[aspect]) > 0)
      .map((aspect) => aspect.toUpperCase());
    if (flagged.length) {
      lines.push(band[1] + ' ' + flagged.join(', '));
    }
  });
  return lines;
};

const args = input || {};
const pathMatch = window.location.pathname.match(/^\/region\/([^/?#]+)/);
const asked = args.region || (pathMatch ? pathMatch[1] : '');
const wanted = normalise(asked);
if (!wanted) {
  return 'No region was given and this is not an NZAA region page. Ask for a ' +
    'region by name, for example "Queenstown", or open ' +
    'https://www.avalanche.net.nz/region/<region>.';
}

try {
  const [regionData, forecastData] = await Promise.all([
    getJson('/api/region'),
    getJson('/api/forecast'),
  ]);

  const regions = (regionData && regionData.regions) || [];
  const region = regions.find((r) => normalise(r.urlSegment) === wanted) ||
    regions.find((r) => normalise(r.title) === wanted) ||
    regions.find((r) => normalise(r.title).indexOf(wanted) === 0);
  if (!region) {
    // "Outside Forecast Region" is the catch-all polygon for public
    // observations logged beyond the forecast areas, so leave it out of the
    // suggestions even though it resolves like any other region.
    return 'Unknown region "' + asked + '". The NZAA forecast regions are: ' +
      regions
        .filter((r) => r.urlSegment !== 'outside-forecast-region')
        .map((r) => r.title)
        .join(', ') + '.';
  }

  const forecasts = ((forecastData && forecastData.forecasts) || [])
    .filter((f) => Number(f.regionId) === Number(region.id));
  if (!forecasts.length) {
    return 'No current avalanche forecast is published for ' + region.title +
      '. The NZAA only forecasts during the winter season.';
  }
  // Timestamps are zero-padded, so a string compare orders them correctly.
  const forecast = forecasts.reduce((a, b) => (a.created > b.created ? a : b));

  const lines = [];
  lines.push('NZ Avalanche Advisory forecast for ' + region.title + '.');
  lines.push('Issued ' + forecast.created + ', valid ' +
    forecast.validPeriod + ' until ' +
    validUntil(forecast.created, forecast.validPeriod) +
    ' (NZ local time). Forecaster: ' + (forecast.forecaster || 'unknown') +
    '.');

  lines.push('');
  lines.push('Danger rating by elevation:');
  (forecast.altitudeDanger || []).forEach((band, index) => {
    lines.push('- ' + bandLabel(band, index) + ': ' +
      ratingLabel(band.rating));
    const advice = toText(band.description);
    if (advice) {
      lines.push('  ' + advice);
    }
  });

  const dangers = (forecast.avalancheDangers || []).slice()
    .sort((a, b) => (a.priority_level || 0) - (b.priority_level || 0));
  if (dangers.length) {
    lines.push('');
    lines.push('Avalanche problems (likelihood and size are 1-5 scales: ' +
      'likelihood 1 unlikely, 3 likely, 5 almost certain; size 1 smallest, ' +
      '5 largest):');
    dangers.forEach((danger) => {
      const character = (danger.character && danger.character.title) ||
        'Unspecified';
      lines.push('- ' + character + ', ' +
        (danger.priority || 'unranked').toLowerCase() + ' problem.');
      const facts = [];
      if (danger.likelihood) {
        facts.push('likelihood ' + danger.likelihood + '/5');
      }
      if (danger.size) {
        facts.push('size ' + danger.size + '/5');
      }
      if (danger.trend) {
        facts.push('trend ' + (TRENDS[danger.trend] ||
          String(danger.trend).toLowerCase()));
      }
      if (danger.time) {
        facts.push(danger.time.isAllDay ? 'all day' :
          'from ' + hourMinute(danger.time.start) + ' to ' +
          hourMinute(danger.time.end));
      }
      const rose = roseLines(danger.aspects);
      if (rose.length) {
        facts.push('aspects ' + rose.join('; '));
      }
      if (facts.length) {
        lines.push('  ' + facts.join(', ') + '.');
      }
      const description = toText(danger.description);
      if (description) {
        lines.push('  ' + description);
      }
    });
  }

  if (forecast.confidenceLevel) {
    lines.push('');
    lines.push('Forecaster confidence: ' + forecast.confidenceLevel + '.');
    (forecast.confidenceReasons || []).forEach((reason) => {
      const text = toText(reason);
      if (text) {
        lines.push('- ' + text);
      }
    });
  }

  const important = toText(forecast.importantInformation);
  if (important) {
    lines.push('');
    lines.push('Important information: ' + important);
  }

  (forecast.additionalInformation || []).forEach((section) => {
    const content = toText(section.content);
    if (content) {
      lines.push('');
      lines.push((section.title || 'Additional information') + ': ' + content);
    }
  });

  lines.push('');
  lines.push('Source: https://www.avalanche.net.nz/region/' +
    region.urlSegment);
  return lines.join('\n');
} catch (error) {
  return 'Could not read the NZ Avalanche Advisory forecast API: ' +
    (error && error.message ? error.message : String(error));
}
