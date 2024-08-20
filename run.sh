#!/bin/sh

# Run experiment 1
python main.py -f config/experiment_1/independent.yaml
python main.py -f config/experiment_1/conditional.yaml

# Run experiment 2
python main.py -f config/experiment_2/prospective.yaml
python main.py -f config/experiment_2/retrospective.yaml

# Run experiment 3a
for i in {0..5}; do
    python main.py -f config/experiment_3a/prospective_$i.yaml
done

for i in {0..5}; do
    python main.py -f config/experiment_3a/retrospective_$i.yaml
done

# Run experiment 3b
for i in {0..5}; do
    python main.py -f config/experiment_3b/prospective_$i.yaml
done

for i in {0..5}; do
    python main.py -f config/experiment_3b/retrospective_$i.yaml
done