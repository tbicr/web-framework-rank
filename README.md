# Web Framework Rank
Wisdom of the crowd web framework rank.

## Rank and Score
name | rank | score | pypistats downloads last month | pypi projects | stackoverflow questions | github stars | repo unique committers | repo changed lines last month | repo unique committers last month | repo last commit
:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:
Django | 1 | 97 | 7630665 | 13813 | 267715 | 57786 | 2509 | 3914 | 40 | 2021-06-04
Flask | 2 | 93 | 25853339 | 2375 | 45740 | 55622 | 739 | 3426 | 16 | 2021-06-02
Tornado | 3 | 81 | 16003135 | 255 | 3659 | 20017 | 426 | 124 | 3 | 2021-05-30
AIOHTTP | 4 | 80 | 24957726 | 190 | 1231 | 11261 | 608 | 258 | 5 | 2021-05-31
Dash | 5 | 79 | 507280 | 556 | 2443 | 14626 | 93 | 77013 | 7 | 2021-06-03
FastAPI | 6 | 76 | 2047616 | 147 | 1200 | 31769 | 230 | 3659 | 7 | 2021-05-26
Twisted | 7 | 76 | 1746110 | 62 | 3386 | 4271 | 262 | 14377 | 7 | 2021-06-01
Sanic | 8 | 75 | 3910933 | 156 | 165 | 15024 | 321 | 1083 | 2 | 2021-06-04
Werkzeug | 9 | 75 | 28068749 | 16 | 561 | 5741 | 428 | 3090 | 8 | 2021-06-01
Zope | 10 | 65 | 26882 | 298 | 715 | 263 | 171 | 1708 | 5 | 2021-06-04
Falcon | 11 | 63 | 606984 | 108 | 182 | 8415 | 178 | 356 | 3 | 2021-05-26
Starlette | 12 | 59 | 2287166 | 56 | 100 | 5599 | 160 | 99 | 4 | 2021-05-27
Pyramid | 13 | 58 | 1198114 | 418 | 2189 | 3562 | 354 | 0 | 0 | 2021-03-15
Bottle | 14 | 55 | 1948133 | 148 | 1467 | 7271 | 220 | 0 | 0 | 2021-01-01
Quart | 15 | 53 | 131412 | 50 | 93 | 903 | 58 | 559 | 2 | 2021-06-01
CherryPy | 16 | 46 | 381441 | 31 | 1352 | 1405 | 139 | 0 | 0 | 2021-05-03
web2py | 17 | 42 | 685 | 8 | 2132 | 1946 | 262 | 0 | 0 | 2021-03-03
web.py | 18 | 40 | 74622 | 7 | 886 | 5568 | 88 | 0 | 0 | 2021-03-03
TurboGears | 19 | 40 | 26358 | 4 | 153 | 760 | 35 | 23 | 1 | 2021-05-26
hug | 20 | 40 | 29352 | 58 | 34 | 6502 | 123 | 0 | 0 | 2020-08-10
Emmett | 21 | 35 | 386 | 4 | 0 | 659 | 21 | 40 | 1 | 2021-06-01
Pylons | 22 | 33 | 88585 | 17 | 834 | 212 | 36 | 0 | 0 | 2018-01-12
Grok | 23 | 32 | 1381 | 88 | 407 | 18 | 40 | 0 | 0 | 2020-09-02
Quixote | 24 | 31 | 135 | 4 | 0 | 70 | 6 | 14 | 2 | 2021-06-01
Morepath | 25 | 27 | 957 | 12 | 0 | 387 | 27 | 0 | 0 | 2021-04-18
Vibora | 26 | 24 | 716 | 1 | 0 | 5718 | 27 | 0 | 0 | 2019-02-11
CubicWeb | 27 | 23 | 3024 | 152 | 0 | 0 | 0 | 0 | 0 | 
Pycnic | 28 | 20 | 1513 | 1 | 0 | 156 | 10 | 0 | 0 | 2021-02-16
Growler | 29 | 18 | 38 | 4 | 0 | 684 | 6 | 0 | 0 | 2020-03-08
Giotto | 30 | 15 | 68 | 7 | 0 | 54 | 3 | 0 | 0 | 2013-10-07

## Fields
Next fields used to calculate score and rank it:
- unique committers
- last update
- lines updated last month
- unique committers last month
- github stars
- stackoverflow questions for appropriate framework tag
- last month downloads
- projects on pypi with framework name substring

## Score Calculation Algorithm
Score calculated as next way:
- calculate field score for framework for each field
  - for each field get framework value
  - order values from smallest to heights
  - field score for framework is index in ordered list starting from 1 divided to count of frameworks
- sum filed score of each field and divide to number of files
- increase score to 100 to get result form 0 to 100
