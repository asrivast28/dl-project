import re
import sys

pattern = re.compile(r'Train Epoch: (\d+).*Train Loss: (\d+.\d+)\s+Val Loss: (\d+.\d+)\s+Val Acc: (\d+)')

if len(sys.argv) < 2:
    print 'Usage: python parse_logs.py <logfile>'
else:
    with open(sys.argv[1], 'rb') as log:
        for line in log.xreadlines():
            match = pattern.match(line)
            if match is not None:
                print '%s,%s,%s,%s'%(match.group(1), match.group(2), match.group(3), match.group(4))
