# Email Verification

Email verification means - check if an email address exists or not without sending an email. This is useful if you manage large email marketing lists of consumer email addresses -- if a bunch of emails you send hard bounce, then you'll be blocked by your ESP. Email verification is a way to avoid that.

## Email Verification Companies

The companies below provide UIs and APIs for verifying emails. Costs are usually in the cents-per-email range, so they're very affordable.

- **Kickbox** - https://kickbox.com (My favorite) - They offer bulk (list) verification for verifying a whole list, and also a real-time verification API for verifying email at the point of capture. The API will also detect disposable email addresses, and suggest spelling corrections if it thinks the user made a typo (eg gmial instead of gmail). I like them because they have a strict no-spammer customer policy, so their results are usually very good.

- **Neverbounce** - https://neverbounce.com - Owned by Zoominfo (an email guessing sales tool), they offer bulk verification and real time verification.

- **ZeroBounce** - https://zerobounce.com - They also offer list verification and API verification.

- **Validity / BriteVerify** - https://validity.com - Briteverify used to be the 'best' offering, but now they're part of the validity suite, which is a whole bunch of email and sales tools bundled together.

## Open source email verification tools

The tools below provide email verification as a self-hosted solution, but due to the nature of how SMTP reputation works, they're probably not going to be super accurate unless you already have [warm IPs](https://blog.kickbox.com/what-is-ip-warming-why-is-it-important/). Plus many email providers just accept all emails without asking questions as an anti-spam measure, so your milage may vary.

- **email-verifier (go)** - https://github.com/AfterShip/email-verifier
- **check-if-email-exists** - https://github.com/reacherhq/check-if-email-exists
- **email-exists** - https://github.com/MarkTiedemann/email-exists



## Attribution

Site logo from [Juicy Fish](https://www.flaticon.com/authors/juicy-fish)
