# CLAUDE.md - Environment Guide for Email Verifier Reviews

Live website: emailverifiers.com
Goal of site: Serve as the #1 resource for people learning about email verification tools (devs, marketers, and pms). We want folks to bookmark the site, and we want good inbound seo traffic.

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


## Cron task instructions

Your job is two things:
1. To write and commit a new blog post for the site.
2. Review the live site and fix any mistakes


### Writing a new post
Our target audience are: marketers, developers, and product managers who are involved in email collection and email marketing.

1. Scan existing blog posts to understand the style and form of the posts we need.
2. Write a new blog post in the same style.
3. Review the blog post to make sure the content is correct, and to verify it will compile to HTML correctly
4. Commit the blog post

### Reviewing the live site
1. You're going to check two blog posts: yesterday's post, and a random historical blog post
2. Fetch them from the live website, review the HTML, look for a) rendering issues, b) broken links, c) invalid information, d) conformity to the guidance provided
3. If there are errors - fix and commit them
4. If no errors - don't do anything

Guidance:
- No more than 20% of an article should be code. Continuous code lines should be under 50.
- Be careful with code blocks and escaping of content within a codeblock
- Liquid syntax in a codeblock requires you to use {% raw %}
- Set the blog post date to be yesterday's date, so that it appears right away
- Blog about email verification, but also blog about topics related to the jobs of our target audience - like email marketing, spam prevention, web development, etc.
- Content is meant to be helpful and guide users to other resources, link out whenever possible.
- Tone should be friendly but clear, like a real expert is writing it.
- Goal is to educate users on useful topics, just be as helpful as possible.
