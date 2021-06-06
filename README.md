# Web Framework Rank
Wisdom of the crowd web framework rank.

## Rank and Score
name | rank | score | pypistats downloads last month | pypi projects | stackoverflow questions | github stars | repo unique committers | repo changed lines last month | repo unique committers last month | repo last commit
:--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:
Django | 1 | 98 | 7516673 | 13822 | 267805 | 57813 | 2510 | 3738 | 40 | 2021-06-05
Flask | 2 | 94 | 25569819 | 2378 | 45749 | 55641 | 739 | 3426 | 16 | 2021-06-02
Tornado | 3 | 82 | 15618091 | 255 | 3659 | 20020 | 426 | 124 | 3 | 2021-05-30
AIOHTTP | 4 | 81 | 24431713 | 190 | 1231 | 11264 | 608 | 258 | 5 | 2021-05-31
Dash | 5 | 79 | 496299 | 557 | 2445 | 14635 | 93 | 52337 | 5 | 2021-06-03
Twisted | 6 | 77 | 1691793 | 62 | 3386 | 4271 | 262 | 12911 | 7 | 2021-06-01
Werkzeug | 7 | 75 | 27700687 | 16 | 561 | 5740 | 428 | 3088 | 7 | 2021-06-01
FastAPI | 8 | 74 | 2003129 | 147 | 1201 | 31813 | 230 | 3340 | 4 | 2021-05-26
Sanic | 9 | 73 | 3835925 | 156 | 165 | 15025 | 321 | 1083 | 2 | 2021-06-04
Zope | 10 | 64 | 26263 | 298 | 715 | 263 | 171 | 1662 | 5 | 2021-06-04
Falcon | 11 | 63 | 584940 | 108 | 182 | 8420 | 178 | 356 | 3 | 2021-05-26
Starlette | 12 | 59 | 2233197 | 56 | 100 | 5604 | 160 | 99 | 4 | 2021-05-27
Pyramid | 13 | 58 | 1227513 | 418 | 2189 | 3562 | 354 | 0 | 0 | 2021-03-15
Bottle | 14 | 55 | 1885581 | 148 | 1468 | 7272 | 220 | 0 | 0 | 2021-01-01
Quart | 15 | 53 | 131762 | 50 | 93 | 906 | 58 | 559 | 2 | 2021-06-01
CherryPy | 16 | 46 | 372371 | 31 | 1352 | 1404 | 139 | 0 | 0 | 2021-05-03
web2py | 17 | 42 | 657 | 8 | 2132 | 1946 | 262 | 0 | 0 | 2021-03-03
web.py | 18 | 40 | 72227 | 7 | 886 | 5570 | 88 | 0 | 0 | 2021-03-03
TurboGears | 19 | 40 | 24906 | 4 | 153 | 761 | 35 | 23 | 1 | 2021-05-26
hug | 20 | 40 | 28272 | 58 | 34 | 6504 | 123 | 0 | 0 | 2020-08-10
Emmett | 21 | 35 | 357 | 4 | 0 | 659 | 21 | 40 | 1 | 2021-06-01
Pylons | 22 | 33 | 84373 | 17 | 834 | 211 | 36 | 0 | 0 | 2018-01-12
Quixote | 23 | 32 | 135 | 4 | 0 | 70 | 6 | 14 | 2 | 2021-06-01
Grok | 24 | 32 | 1374 | 88 | 407 | 18 | 40 | 0 | 0 | 2020-09-02
Morepath | 25 | 27 | 922 | 12 | 0 | 387 | 27 | 0 | 0 | 2021-04-18
Vibora | 26 | 24 | 678 | 1 | 0 | 5719 | 27 | 0 | 0 | 2019-02-11
CubicWeb | 27 | 23 | 3037 | 152 | 0 | 0 | 0 | 0 | 0 | 
Pycnic | 28 | 20 | 1411 | 1 | 0 | 156 | 10 | 0 | 0 | 2021-02-16
Growler | 29 | 18 | 39 | 4 | 0 | 684 | 6 | 0 | 0 | 2020-03-08
Giotto | 30 | 15 | 66 | 7 | 0 | 54 | 3 | 0 | 0 | 2013-10-07

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
