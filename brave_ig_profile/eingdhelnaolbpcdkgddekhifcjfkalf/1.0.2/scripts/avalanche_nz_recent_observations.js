// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        recent_observations
// @match       https://www.avalanche.net.nz/*
// @description List the recent public field observations submitted to the New
// @description Zealand Avalanche Advisory: observed avalanches (size, trigger,
// @description aspect, elevation, people involved) and snowpack observations
// @description (test results, surface conditions), newest first. Optionally
// @description filter by region, by observation type, and by how many days
// @description back to look.
// @schema      {"type":"object","properties":{"days":{"type":"integer","minimum":1,"maximum":365,"description":"How many days back to search. Defaults to 14."},"region":{"type":"string","description":"Restrict to one NZAA region, for example \"Craigieburn Range\" or \"queenstown\". Omit for all regions."},"type":{"type":"string","enum":["Avalanche","Snowpack"],"description":"Restrict to observed avalanches or to snowpack observations. Omit for both."}}}
// ==/WebMCP==

// The observations page plots each report as a pin on a Mapbox canvas and only
// renders the detail of whichever pin is selected, so the list cannot be read
// from the DOM. Use the site's own unauthenticated, same-origin JSON API:
// /api/observation accepts dateFrom and dateTo (inclusive, YYYY-MM-DD) and
// returns every report in that window, while /api/region supplies the titles
// for the region_id each report carries.

const MAX_REPORTS = 25;
const MAX_COMMENT_CHARS = 500;

const getJson = async (path) => {
  const response = await fetch(path, {
    headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    throw new Error('GET ' + path + ' returned HTTP ' + response.status);
  }
  return response.json();
};

const normalise = (value) =>
  String(value === undefined || value === null ? '' : value)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '');

const collapse = (value) => String(value || '').replace(/\s+/g, ' ').trim();

const truncate = (value, limit) => {
  const text = collapse(value);
  if (text.length <= limit) {
    return text;
  }
  return text.slice(0, limit - 1).trim() + '\u2026';
};

// The API compares its date filters against NZ wall-clock timestamps. A day of
// slop either side does not matter for "recent observations", so the viewer's
// local date is a good enough anchor.
const isoDate = (date) => {
  const pad = (value) => String(value).padStart(2, '0');
  return date.getFullYear() + '-' + pad(date.getMonth() + 1) + '-' +
    pad(date.getDate());
};

const args = input || {};
const days = Math.min(Math.max(parseInt(args.days, 10) || 14, 1), 365);
const wantedRegion = normalise(args.region);
const wantedType = normalise(args.type);

const today = new Date();
const from = new Date(today.getTime() - (days - 1) * 86400000);
// Nudge dateTo a day past today: the API filters against NZ wall-clock times,
// so a viewer behind NZ would otherwise miss reports already filed "tomorrow".
const to = new Date(today.getTime() + 86400000);

try {
  const [regionData, observationData] = await Promise.all([
    getJson('/api/region'),
    getJson('/api/observation?dateFrom=' + isoDate(from) + '&dateTo=' +
      isoDate(to)),
  ]);

  const regions = (regionData && regionData.regions) || [];
  const titles = {};
  regions.forEach((region) => {
    titles[Number(region.id)] = region.title;
  });

  // A report logged outside every region polygon carries region_id 0, which
  // the site itself renders as a dash. Fall back to the coordinates, as they
  // are then the only thing locating the report.
  const placeOf = (observation) => {
    const title = titles[Number(observation.region_id)];
    if (title) {
      return title;
    }
    const at = observation.coordinates;
    return at && at.lat && at.lng ?
      'no region (' + at.lat + ', ' + at.lng + ')' : 'no region recorded';
  };

  let region = null;
  if (wantedRegion) {
    region = regions.find((r) => normalise(r.urlSegment) === wantedRegion) ||
      regions.find((r) => normalise(r.title) === wantedRegion) ||
      regions.find((r) => normalise(r.title).indexOf(wantedRegion) === 0);
    if (!region) {
      return 'Unknown region "' + args.region + '". The NZAA regions are: ' +
        regions
          .filter((r) => r.urlSegment !== 'outside-forecast-region')
          .map((r) => r.title)
          .join(', ') + '.';
    }
  }

  const observations = ((observationData && observationData.observations) || [])
    .filter((observation) =>
      !region || Number(observation.region_id) === Number(region.id))
    .filter((observation) =>
      !wantedType || normalise(observation.type) === wantedType)
    // Timestamps are zero-padded, so a string compare orders them correctly.
    .sort((a, b) => String(b.time).localeCompare(String(a.time)));

  const scope = (region ? region.title : 'all NZAA regions') +
    (args.type ? ', ' + args.type + ' reports only' : '');
  if (!observations.length) {
    return 'No public observations were submitted for ' + scope +
      ' in the last ' + days + ' days.';
  }

  const lines = [];
  lines.push(observations.length + ' public observation' +
    (observations.length === 1 ? '' : 's') + ' for ' + scope +
    ' in the last ' + days + ' days, newest first. These are unverified ' +
    'public reports, not forecasts. Times are NZ local.');

  observations.slice(0, MAX_REPORTS).forEach((observation) => {
    const facts = [];
    if (observation.character) {
      facts.push('character ' + observation.character);
    }
    if (observation.elevation) {
      facts.push('elevation ' + observation.elevation + 'm');
    }
    if (observation.aspect) {
      facts.push('aspect ' + observation.aspect);
    }
    if (observation.slopeAngle) {
      facts.push('slope ' + observation.slopeAngle + ' degrees');
    }
    if (observation.size) {
      facts.push('size ' + observation.size);
    }
    if (observation.depth) {
      // "depth" is overloaded: the crown depth of an avalanche, or the total
      // snowpack depth of a pit. The site relabels it by type, so do the same.
      facts.push((observation.type === 'Snowpack' ? 'snowpack depth ' :
        'depth ') + observation.depth + 'cm');
    }
    if (observation.width || observation.length) {
      facts.push('dimensions ' + (observation.width || '?') + 'm wide by ' +
        (observation.length || '?') + 'm long');
    }
    if (observation.triggerType) {
      facts.push('trigger ' + observation.triggerType);
    }
    if (observation.surfaceConditions) {
      facts.push('surface ' + observation.surfaceConditions);
    }
    if (observation.testType && observation.testResult) {
      facts.push('test ' + observation.testType + ' ' +
        collapse(observation.testResult));
    }

    const people = observation.peopleInvolved || {};
    if (people.groupSize || people.peopleCaught || people.peopleInjured) {
      facts.push('party of ' + (people.groupSize || '?') +
        (people.activityType ? ' ' + String(people.activityType).toLowerCase()
          : '') + ', ' + (people.peopleCaught || 0) + ' caught and ' +
        (people.peopleInjured || 0) + ' injured');
    }

    lines.push('');
    lines.push('- ' + String(observation.time).slice(0, 16) + ', ' +
      placeOf(observation) + ', ' + (observation.type || 'observation') +
      ': ' + (collapse(observation.description) || 'no summary given'));
    if (facts.length) {
      lines.push('  ' + facts.join(', ') + '.');
    }
    const comments = truncate(observation.comments, MAX_COMMENT_CHARS);
    if (comments) {
      lines.push('  Observer notes: ' + comments);
    }
    lines.push('  By ' + ((observation.observer && observation.observer.name) ||
      'anonymous') + '. https://www.avalanche.net.nz/observation/' +
      observation.id);
  });

  if (observations.length > MAX_REPORTS) {
    lines.push('');
    lines.push('Showing the ' + MAX_REPORTS + ' most recent of ' +
      observations.length + '. Narrow the window with "days" or "region", or ' +
      'see https://www.avalanche.net.nz/observations');
  }

  return lines.join('\n');
} catch (error) {
  return 'Could not read the NZ Avalanche Advisory observations API: ' +
    (error && error.message ? error.message : String(error));
}
