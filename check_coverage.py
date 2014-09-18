import coverage

coverage.main(['run', '--source', 'minipnm', 'test_minipnm.py'])
coverage.main(['report', '-m'])
coverage.main(['erase'])
