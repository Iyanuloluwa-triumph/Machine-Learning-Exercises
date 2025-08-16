__author__ = "Iyanucodez"


# beginner brainfuck


def brain_luck(code, program_input):
    cells = [0 for x in range(30000)]
    cell_active = 0
    inputs = code.split()
    for item in inputs:
        if item == ">":
            cell_active += 1
        elif item == "<":
            cell_active -= 1
        elif item == "+":
            cells[cell_active] += 1
        elif item == "-":
            cells[cell_active] -= 1
        elif item == ".":
            print(cells[cell_active])
        elif item == ",":
            storage = program_input[0]
            cells[cell_active] = storage
        elif item == "[":
            count = 0
            for _ in cells[item:]:
                count += 1
                while True:
                    if item == ">":
                        cell_active += 1
                    elif item == "<":
                        cell_active -= 1
                    elif item == "+":
                        cells[cell_active] += 1
                    elif item == "-":
                        cells[cell_active] -= 1
                    elif item == ".":
                        print(cells[cell_active])
                    elif item == ",":
                        storage = program_input[0]
                        cells[cell_active] = storage
                    elif item == "]":
                        cell_active += count
                        break

    pass
# copyright brainfuck
