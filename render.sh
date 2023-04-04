
for file in ./checkpoints/*.json; do
  python3 test.py "$file" --device cuda 
done

