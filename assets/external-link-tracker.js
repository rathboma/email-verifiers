(function() {
  'use strict';

  // Function to extract domain from URL
  function getDomain(url) {
    try {
      const urlObj = new URL(url);
      return urlObj.hostname;
    } catch (e) {
      return null;
    }
  }

  // Function to add ref parameter to URL
  function addRefParameter(url) {
    try {
      const urlObj = new URL(url);
      urlObj.searchParams.set('ref', 'emailverifiers.com');
      return urlObj.toString();
    } catch (e) {
      return url; // Return original URL if parsing fails
    }
  }

  function addTrackingParameter(url, domain) {
    const referralCodes = [
      { domain: 'emaillistverify.com', key: 'red', code: 'rathbo'},
      { domain: 'usebouncer.com', newUrl: 'https://withlove.usebouncer.com/ptpivxlk1iee'}
    ]

    const found = referralCodes.find(({domain}) => domain === domain)
    if (!found) return url
    // some sites need a totally new link
    if (found.newUrl) return found.newUrl
    try {
      const urlObj = new URL(url)
      urlObj.searchParams.set(found.key, found.code)
      return urlObj.toString()
    } catch (e) {
      return url
    }
  }

  // Function to handle external link clicks
  function handleExternalLinkClick(event) {
    const link = event.currentTarget;
    const originalUrl = link.href;
    const domain = getDomain(originalUrl);

    if (!domain) return;

    // Fire Fathom tracking event
    if (window.fathom && typeof window.fathom.trackEvent === 'function') {
      window.fathom.trackEvent('click-' + domain);
    }

    // Modify the URL with ref parameter
    const modifiedUrl = addTrackingParameter(addRefParameter(originalUrl), domain);

    // Update the href and allow the click to proceed
    link.href = modifiedUrl;

    // Optional: Log for debugging (remove in production)
    console.log('External link tracked:', {
      domain: domain,
      original: originalUrl,
      modified: modifiedUrl
    });
  }

  // Initialize tracking when DOM is loaded
  function initializeExternalLinkTracking() {
    const trackedLinks = document.querySelectorAll('a[data-track="true"]');

    trackedLinks.forEach(function(link) {
      link.addEventListener('click', handleExternalLinkClick);
    });

    console.log('External link tracking initialized for', trackedLinks.length, 'links');
  }

  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeExternalLinkTracking);
  } else {
    initializeExternalLinkTracking();
  }

  // Re-initialize if content is dynamically loaded
  window.reinitializeExternalLinkTracking = initializeExternalLinkTracking;

})();