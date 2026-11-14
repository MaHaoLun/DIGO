"""Exercise conditional repair on a graph generated in temporary storage."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from native import Native

def main(library):
    with TemporaryDirectory() as tmp:
        graph = Path(tmp) / 'graph'
        graph.write_text('3 2 2 900\n10 0\n20 1\n30 2\n10 20 3 3\n20 30 5 5\n')
        solver = Native(graph, library)
        try:
            mask = [0] * 4096
            mask[0] = mask[2] = 1
            result = solver.query(10, 30, 0, 900, 4, mask)
            assert result['repaired'] and result['cost'] == '8'
            assert result['route'] == [10, 20, 30] and result['departure'] == '0'
            print('Conditional repair, route recovery and earliest departure passed.')
        finally:
            solver.close()

if __name__ == '__main__':
    main(sys.argv[1])
