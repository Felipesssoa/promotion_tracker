// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        danger_ratings
// @match       https://www.avalanche.net.nz/*
// @description Summarise the current avalanche danger rating for every New
// @description Zealand Avalanche Advisory forecast region, north to south, with
// @description the rating for each elevation band and the primary avalanche
// @description problem. Use this to compare regions; use region_forecast for
// @description the full detail of one region.
// @schema      {"type":"object","properties":{}}
// ==/WebMCP==

// The regions on the home page are drawn as coloured polygons on a Mapbox
// canvas, so their danger ratings are not in the DOM at all. Read the site's
// own unauthenticated, same-origin JSON API instead: /api/forecast returns the
// two most recent forecasts for every region (newest first) and /api/region
// supplies the titles and the site's own north-to-south sortOrder. The map
// colours a region by the highest rating across its elevation bands, so do the
// same here for the headline number.

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

const getJson = async (path) => {
  const response = await fetch(path, {
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw new Error('GET ' + path + ' returned HTTP ' + response.status);
  }
  return response.json();
};

const ratingLabel = (rating) => {
  const label = DANGER_RATINGS[String(rating)];
  return (label || 'Unknown') + ' (' + rating + ')';
};

// Each altitudeDanger entry carries only its boundaries; the band it describes
// comes from its position in the array, matching the three rows the site's
// AltitudeDangerRow component renders.
const bandLabel = (band, index) => {
  if (index === 0) {
    return 'above ' + band.altitudeFrom + 'm';
  }
  if (index === 1) {
    return band.altitudeTo + '-' + band.altitudeFrom + 'm';
  }
  return 'below ' + band.altitudeFrom + 'm';
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

try {
  const [regionData, forecastData] = await Promise.all([
    getJson('/api/region'),
    getJson('/api/forecast'),
  ]);

  const forecasts = (forecastData && forecastData.forecasts) || [];
  // Keep only the newest forecast per region. Timestamps are zero-padded, so a
  // string compare orders them correctly.
  const latest = {};
  forecasts.forEach((forecast) => {
    const id = Number(forecast.regionId);
    if (!latest[id] || forecast.created > latest[id].created) {
      latest[id] = forecast;
    }
  });

  const regions = ((regionData && regionData.regions) || [])
    // "Outside Forecast Region" is the catch-all polygon for public
    // observations logged beyond the forecast areas; it is never forecast.
    .filter((region) => region.urlSegment !== 'outside-forecast-region')
    .slice()
    .sort((a, b) => (a.sortOrder || 0) - (b.sortOrder || 0));

  const forecast = [];
  const unforecast = [];
  regions.forEach((region) => {
    if (latest[Number(region.id)]) {
      forecast.push(region);
    } else {
      unforecast.push(region.title);
    }
  });

  if (!forecast.length) {
    return 'No NZ Avalanche Advisory region has a current forecast. The NZAA ' +
      'only forecasts during the winter season.';
  }

  const lines = [];
  lines.push('Current NZ Avalanche Advisory danger ratings, north to south. ' +
    'The headline rating is the highest of the three elevation bands, which ' +
    'is what the map colour shows. Times are NZ local.');

  forecast.forEach((region) => {
    const current = latest[Number(region.id)];
    const bands = current.altitudeDanger || [];
    const peak = bands.reduce((highest, band) =>
      Math.max(highest, Number(band.rating) || 0), 0);
    lines.push('');
    lines.push(region.title + ': ' + ratingLabel(peak));
    if (bands.length) {
      lines.push('  ' + bands
        .map((band, index) =>
          bandLabel(band, index) + ' ' + ratingLabel(band.rating))
        .join(', '));
    }
    const problems = (current.avalancheDangers || []).slice()
      .sort((a, b) => (a.priority_level || 0) - (b.priority_level || 0))
      .map((danger) => (danger.character && danger.character.title) || null)
      .filter(Boolean);
    if (problems.length) {
      lines.push('  Problems: ' + problems.join(', ') + '.');
    }
    lines.push('  Issued ' + current.created + ', valid until ' +
      validUntil(current.created, current.validPeriod) +
      '. Confidence: ' + (current.confidenceLevel || 'unstated') + '.');
    lines.push('  https://www.avalanche.net.nz/region/' + region.urlSegment);
  });

  if (unforecast.length) {
    lines.push('');
    lines.push('No current forecast for: ' + unforecast.join(', ') + '.');
  }

  return lines.join('\n');
} catch (error) {
  return 'Could not read the NZ Avalanche Advisory forecast API: ' +
    (error && error.message ? error.message : String(error));
}
