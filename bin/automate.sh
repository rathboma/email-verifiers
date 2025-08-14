

claude -p "Look at existing posts in this blog and write a new blog post you think will rank well in Google. Make sure it is not a duplicate of another post. Ensure it has the same tone and format as other posts, also make sure it links to another relevent post in a natural way. When you're done, review the post for correctness, then commit the new content and push it to the main branch." \
       --append-system-prompt "You are an expert in email marketing and technical email marketing topics. You are also an SEO marketing expert" \
       --allowedTools "Bash(git:*)" "Edit" \
       --permission-mode acceptEdits
