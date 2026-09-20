"""Data sources. Two public APIs, two different prices.

- ``comprasgov``: unit price actually PAID, per item, after the auction.
  Training data.
- ``pncp``: what the agency ESTIMATED before bidding. The number to audit.

Every puller is resumable and appends as it goes: a crash or a rate limit
costs at most one page.
"""
