// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        page_heading
// @match       https://example.com/*
// @description Return the text of the first top-level heading (<h1>) on the
// @description current page.
// @schema      {"type":"object","properties":{}}
// ==/WebMCP==

// Generic example used for easy local verification on example.com.
const h1 = document.querySelector('h1');
return h1 ? h1.textContent.trim() : 'No <h1> found on this page.';
