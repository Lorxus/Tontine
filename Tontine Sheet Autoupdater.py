import ezsheets

full = ezsheets.Spreadsheet('1scomCAeojAeMXYI4x7dtftkxXGdvG7g8Kj7MWulscU8')
first = full[0]
stats = full[7]

first.refresh()
# the JSON might need updating

day = int(first['B5'])
row = day + 2
# offset between days and rows

pop = int(first['A18'])
stats[3, row] = pop
# set today's pop to the correct value