import ast


def list_classes_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    class_names = [
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    ]

    return class_names


# Example usage
if __name__ == "__main__":
    file_path = "/Users/josephmargaryan/Desktop/quantbayes/quantbayes/dmm/test_integration.py"  # Replace with your file path
    classes = list_classes_from_file(file_path)
    print("Classes defined in the file:", classes)
