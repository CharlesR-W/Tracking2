from __future__ import annotations

import copy

import torch
from torch.nn import functional as F

from tracking2.criticality import transplant_module
from tracking2.models import InstrumentedVGG19


torch.manual_seed(7)
torch.set_num_threads(2)
model = InstrumentedVGG19(classifier_width=16, width_multiplier=0.0625)
x = torch.zeros(16, 3, 32, 32)
y = torch.cat([torch.zeros(8), torch.ones(8)]).long()
x[8:] = 1.0
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

model.train()
initial_loss = float(F.cross_entropy(model(x), y).detach())
for _ in range(120):
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    trained_logits = model(x)
    trained_loss = float(F.cross_entropy(trained_logits, y))
    trained_accuracy = float((trained_logits.argmax(1) == y).float().mean())

torch.manual_seed(7007)
random_source = InstrumentedVGG19(classifier_width=16, width_multiplier=0.0625)
reset = copy.deepcopy(model)
transplant_module(reset, random_source, 18)
reset.eval()
with torch.no_grad():
    reset_loss = float(F.cross_entropy(reset(x), y))
    reset_accuracy = float((reset(x).argmax(1) == y).float().mean())

assert trained_loss < initial_loss * 0.30, (initial_loss, trained_loss)
assert trained_accuracy >= 0.90, trained_accuracy
assert reset_loss > trained_loss + 0.5, (trained_loss, reset_loss)
print({
    "initial_loss": initial_loss,
    "trained_loss": trained_loss,
    "trained_accuracy": trained_accuracy,
    "reset_loss": reset_loss,
    "reset_accuracy": reset_accuracy,
})
