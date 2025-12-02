(function() {
  'use strict';

  function maybeTrack(...parts) {
    if (window.fathom && typeof window.fathom.trackEvent === 'function') {
      window.fathom.trackEvent(parts.join("-"));
    }
  }

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
      { domain: 'usebouncer.com', newUrl: 'https://withlove.usebouncer.com/ptpivxlk1iee'},
      { domain: 'zerobounce.com', newUrl: 'https://aff.zerobounce.net/e19rG6'},
      { domain: 'kickbox.com', newUrl: 'https://kickbox.com?fp_ref=evcom'},
    ]

    try {
      const found = referralCodes.find(({domain: d}) => domain.includes(d))
      if (!found) {
        return url
      }

      let result = null
      // some sites need a totally new link
      if (found.newUrl) {
        result = found.newUrl
      }
      const urlObj = new URL(url)
      urlObj.searchParams.set(found.key, found.code)
      result = urlObj.toString()

      if (result) {
        maybeTrack('referral', domain)
        return result
      } else {
        return url
      }
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

    maybeTrack('click', domain)
    // Fire Fathom tracking event


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