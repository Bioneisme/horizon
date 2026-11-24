import re, sys
src, level, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lines, keep, skip = open(src).read().split('\n'), [], 0
for line in lines:
    m = re.match(r'\s*//@(\d)\{$', line)
    if m:
        if skip or int(m.group(1)) > level:
            skip += 1
        continue
    if re.match(r'\s*//@\}$', line):
        if skip:
            skip -= 1
        continue
    if not skip:
        keep.append(line)
open(out, 'w').write('\n'.join(keep))
