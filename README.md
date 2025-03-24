# Email Verifier Reviews

A comprehensive directory and review site for email verification services, built with Jekyll and Tailwind CSS.

## Site Structure

- `/` - Home page with featured services and information
- `/services/` - Directory of all email verification services
- `/comparisons/` - Comparison articles between different services
- `/about/` - Information about the site

## Collections

- `_services` - Individual service pages with reviews
- `_comparisons` - Comparison articles between services

## Development

### Prerequisites

- Ruby 3.0+
- Bundler
- Jekyll

### Setup

1. Clone the repository
2. Run `bundle install`
3. Start the development server with `./serve.sh` or `bundle exec jekyll serve --livereload`

### Adding Content

#### New Service

Create a new Markdown file in the `_services` directory with the following front matter:

```yaml
---
layout: service
title: Service Name
website: https://example.com
rating: 4.5
excerpt: Short description of the service.
offers_bulk: true
offers_api: true
offers_integrations: true
starting_price: $X.XX per email
best_for: Description of ideal use case
slug: service-name
pros:
  - Pro point 1
  - Pro point 2
  - Pro point 3
cons:
  - Con point 1
  - Con point 2
verdict: Overall assessment of the service.
pricing: |
  Detailed pricing information in markdown format.
---

Content of the review in markdown.
```

#### New Comparison

Create a new Markdown file in the `_comparisons` directory with the following front matter:

```yaml
---
layout: comparison
title: Comparison Title
description: Brief description of the comparison.
services: [service1-slug, service2-slug]
recommendation: Your recommendation text here.
slug: comparison-slug
---

Content of the comparison in markdown.
```

### Building for Production

To build the site for production, run:

```bash
bundle exec jekyll build
```

The built site will be in the `_site` directory.