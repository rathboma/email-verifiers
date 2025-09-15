require 'nokogiri'
require 'uri'

module Jekyll
  class ExternalLinkTracker
    def self.process(content, site_url)
      doc = Nokogiri::HTML::DocumentFragment.parse(content)
      site_uri = URI.parse(site_url)

      doc.css('a[href]').each do |link|
        href = link['href']
        next if href.nil? || href.empty?

        # Skip internal links, anchors, mailto, tel, etc.
        next if href.start_with?('#', 'mailto:', 'tel:', 'javascript:', '/')
        next if href.start_with?(site_url)

        # Parse the URL to check if it's external
        begin
          link_uri = URI.parse(href)

          # Only process http/https links
          if link_uri.scheme && ['http', 'https'].include?(link_uri.scheme.downcase)
            # Check if it's truly external (handle www subdomain variations)
            if link_uri.host && !self.same_domain?(link_uri.host, site_uri.host)
              link['data-track'] = 'true'
            end
          end
        rescue URI::InvalidURIError
          # Skip malformed URLs
          next
        end
      end

      doc.to_html
    end

    # Helper method to compare domains, handling www subdomain variations
    def self.same_domain?(host1, host2)
      # Normalize hosts by removing www prefix
      normalized_host1 = host1.downcase.sub(/^www\./, '')
      normalized_host2 = host2.downcase.sub(/^www\./, '')

      normalized_host1 == normalized_host2
    end
  end

  # Hook into Jekyll's post_render to process all content
  Jekyll::Hooks.register [:posts, :pages, :documents], :post_render do |item|
    if item.output_ext == '.html'
      item.output = ExternalLinkTracker.process(item.output, item.site.config['url'])
    end
  end
end