

claude -p "Follow the cron task instructions in the CLAUDE.md file" \
       --append-system-prompt "You are an expert in email marketing and technical email marketing topics. You are also an SEO marketing expert" \
       --allowedTools "Bash(git:*)" "Edit" "WebFetch(domain:emailverifiers.com)" \
       --permission-mode acceptEdits
