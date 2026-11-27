"""Emergency fallback: output the topology of the first input gene tree (binary, same labels)."""
import re, sys
text = open(sys.argv[1]).read()
tree = text[:text.index(';')]
tree = re.sub(r'\)[^(),:;]*', ')', tree)        # drop internal labels
tree = re.sub(r':[^(),:;]*', '', tree)          # drop branch lengths
open(sys.argv[2], 'w').write(''.join(tree.split()) + ';\n')
