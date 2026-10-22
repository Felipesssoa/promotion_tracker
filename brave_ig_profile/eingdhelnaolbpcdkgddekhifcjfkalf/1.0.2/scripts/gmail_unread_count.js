// Copyright (c) 2026 The Brave Authors. All rights reserved.
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

// ==WebMCP==
// @name        unread_count
// @match       https://mail.google.com/*
// @description Report the number of unread messages in the Gmail inbox, read
// @description from the bold count next to the Inbox label in the left
// @description navigation. Reads the DOM rather than any Gmail API.
// @schema      {"type":"object","properties":{}}
// ==/WebMCP==

// The body below runs as the async execute(input) callback in the page's main
// world. It must `return` a string (or a value coercible to one).

// The Inbox nav link always points at the #inbox view, so match on that rather
// than on Gmail's volatile CSS class names. Its aria-label reads e.g.
// "Inbox 2112 unread", which is exactly the information we want to surface.
const inboxLink = document.querySelector('a[href*="#inbox"]');
if (!inboxLink) {
  return 'Could not find the Inbox link on this page.';
}
return inboxLink.getAttribute('aria-label') || 'Inbox 0 unread';
