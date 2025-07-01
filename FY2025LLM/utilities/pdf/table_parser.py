"""
PDF 에서 표를 파싱하는 클래스

"""
class PDFTableParser:
    def parse_tables(self, layout) -> List[Dict]:
        tables = []
        for t in layout.tables:
            table = self._parse_single_table(t)
            tables.append(table)
        return tables

    def _parse_single_table(self, table) -> Dict:
        matrix = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
        for c in table.cells:
            for rs in range(c.row_span):
                for cs in range(c.column_span):
                    content = c.content.replace("•", "").replace("unselected:", "")
                    if c.kind == "columnHeader" and c.column_span > 1:
                        content += "_concat"
                    matrix[c.row_index + rs][c.column_index + cs] = content
        return {
            "offset": table.spans[0].offset,
            "matrix": matrix,
            "bounding": table.bounding_regions
        }
