# most abstracts have © at the end; remove this for cleaner corpora
# performed once on txt file of abstracts; then put cleaned file into csv
# not an issue for LLS data, only for scopus

with open('abstracts-raw.txt', 'r') as file:
    s = file.readlines()
new_s = [re.sub(r"\©.*$", "", line) for line in s]

with open('abstracts-cleaned.txt', 'w') as f:
    for line in new_s:
        f.write(f"{line}")