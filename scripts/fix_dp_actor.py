#!/usr/bin/env python3
content = open('/root/DeepResearcher/verl/workers/actor/dp_actor.py').read()
old = '        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)'
new = '''        # self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)  # Disabled to avoid compile workers hang
        self.compute_entropy_from_logits = verl_F.entropy_from_logits'''
if old in content:
    content = content.replace(old, new)
    open('/root/DeepResearcher/verl/workers/actor/dp_actor.py', 'w').write(content)
    print('Modified OK')
else:
    print('Not found')
    print('Looking for:', repr(old[:80]))
