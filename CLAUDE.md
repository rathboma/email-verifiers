# CLAUDE.md - Environment Guide for Email Verifier Reviews

## Development Commands
- Start local server: `bundle exec jekyll serve --livereload` or `./serve.sh`
- Build for production: `bundle exec jekyll build`
- Install dependencies: `bundle install`

## Content Guidelines
- Service files (.md) go in `_services/` directory
- Comparison files go in `_comparisons/` directory
- Follow YAML frontmatter structure in README.md examples
- Use descriptive slugs that match title (lowercase, hyphenated)

## Style Guidelines
- Use Markdown for content formatting
- Keep reviews objective and factual
- Structure content with clear headings (## and ###)
- Include all required YAML frontmatter fields
- For pricing sections, use the pipe operator (|) for multi-line content
- Pros/cons should be concise bullet points (3-5 each)
- Maintain consistent rating scale (0.0-5.0)

## Technical Notes
- This is a Jekyll site with Tailwind CSS
- Ruby 3.0+ required
- Generated site appears in `_site/` directory