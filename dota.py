import json
import bz2

with bz2.BZ2File('./matches.jsonlines.bz2') as matches_file:
    for line in matches_file:
        match = json.loads(line)

        # Обработка матча
        break

