from graphviz import Digraph


def create_rmdp_flowchart():
    # Create a new directed graph
    dot = Digraph(comment="RMDP Solver Algorithm")

    # Adjust layout settings
    dot.attr(
        rankdir="TB",  # Top to bottom direction
        nodesep="0.3",  # Reduce vertical space between nodes
        ranksep="0.2",  # Reduce horizontal space between ranks
        splines="ortho",
    )  # Use orthogonal lines

    # Adjust node attributes for more compact appearance
    dot.attr("node", shape="box", style="rounded", height="0.4", width="0.4", fontsize="10")
    # Add nodes with different shapes/colors
    # Process nodes (rectangles)
    nodes_process = {
        "A": "Start",
        "B": "Initialize:\nempty best decision\nmax delay, zero slack",
        "C": "Generate all potential\norder sequences",
        "E": "Create candidate\nroute plan",
        "F": "Initialize empty\npostponement set",
        "H": "Find best vehicle",
        "I": "Assign order\nto vehicle",
        "K": "Add to\npostponement set",
        "L": "Skip postponement",
        "M": "Create candidate\ndecision",
        "O": "Update best solution:\ndecision, delay, slack",
        "P": "Keep current best",
        "S": "Remove postponed orders\nfrom final route plan",
        "T": "Return final route plan\nand postponed orders",
        "U": "End",
    }

    # Decision nodes (diamonds)
    nodes_decision = {
        "D": "For each\nsequence",
        "G": "For each order\nin sequence",
        "J": "Can order be\npostponed?",
        "N": "Is solution better?\nLower delay OR\nEqual delay with less slack",
        "Q": "More orders?",
        "R": "More sequences?",
    }

    # Add process nodes
    for id, label in nodes_process.items():
        dot.node(id, label)

    # Add decision nodes with diamond shape
    for id, label in nodes_decision.items():
        dot.node(id, label, shape="diamond")

    # Add edges
    edges = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("D", "E"),
        ("E", "F"),
        ("F", "G"),
        ("G", "H"),
        ("H", "I"),
        ("I", "J"),
        ("J", "K", "Yes"),
        ("J", "L", "No"),
        ("K", "M"),
        ("L", "M"),
        ("M", "N"),
        ("N", "O", "Yes"),
        ("N", "P", "No"),
        ("O", "Q"),
        ("P", "Q"),
        ("Q", "G", "Yes"),
        ("Q", "R", "No"),
        ("R", "D", "Yes"),
        ("R", "S", "No"),
        ("S", "T"),
        ("T", "U"),
    ]

    # Add edges with labels
    for edge in edges:
        if len(edge) == 2:
            dot.edge(edge[0], edge[1])
        else:
            dot.edge(edge[0], edge[1], edge[2])

    # Save the graph
    dot.render("rmdp_flowchart", format="png", cleanup=True)

    print("Flowchart has been created as 'rmdp_flowchart.png'")


if __name__ == "__main__":
    create_rmdp_flowchart()

# graph TD
#     A[Start] --> B[Initialize empty best decision, max delay, zero slack]
#     B --> C[Generate all potential order sequences]
#     C --> D{For each sequence}
#     D --> E[Create candidate route plan]
#     E --> F[Initialize empty postponement set]
#     F --> G{For each order in sequence}
#     G --> H[Find best vehicle]
#     H --> I[Assign order to vehicle]
#     I --> J{Can order be postponed?}
#     J -->|Yes| K[Add to postponement set]
#     J -->|No| L[Skip postponement]
#     K --> M[Create candidate decision]
#     L --> M
#     M --> N{Is solution better?<br>Lower delay OR<br>Equal delay with less slack}
#     N -->|Yes| O[Update best solution:<br>decision, delay, slack]
#     N -->|No| P[Keep current best]
#     O --> Q{More orders?}
#     P --> Q
#     Q -->|Yes| G
#     Q -->|No| R{More sequences?}
#     R -->|Yes| D
#     R -->|No| S[Remove postponed orders<br>from final route plan]
#     S --> T[Return final route plan<br>and postponed orders]
#     T --> U[End]
