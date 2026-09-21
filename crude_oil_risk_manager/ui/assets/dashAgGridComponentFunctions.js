// Custom AG Grid cell renderers for the Structures tab.
// Colours arrive through cellRendererParams (from COLORS in ui/layouts/shell.py),
// so the palette stays defined in one place. Clicks report through props.setData,
// which dash-ag-grid exposes to Dash as the grid's `cellRendererData` property.

var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});

// Structure name; clicking it asks for the detail view.
dagcomponentfuncs.StructureNameLink = function (props) {
    var colors = props.colors || {};
    return React.createElement(
        "a",
        {
            href: "#",
            onClick: function (event) {
                event.preventDefault();
                props.setData({ action: "view", structure_id: props.data.structure_id });
            },
            style: { color: colors.ACCENT_GREEN, fontWeight: "bold", textDecoration: "none", cursor: "pointer" },
        },
        props.value
    );
};

// SHELL = grey badge, OPEN = green badge.
dagcomponentfuncs.StatusBadge = function (props) {
    var colors = props.colors || {};
    var isOpen = props.value === "OPEN";
    var color = isOpen ? colors.ACCENT_GREEN : colors.TEXT_SECONDARY;
    return React.createElement(
        "span",
        {
            style: {
                color: color,
                border: "1px solid " + color,
                borderRadius: "10px",
                padding: "2px 10px",
                fontSize: "11px",
                fontWeight: "bold",
                letterSpacing: "0.05em",
            },
        },
        props.value
    );
};

// Row actions: shell -> Enter Trade + View, open -> View + Exit.
dagcomponentfuncs.StructureActions = function (props) {
    var colors = props.colors || {};
    var status = props.data.status;

    function button(label, action, color) {
        return React.createElement(
            "button",
            {
                key: action,
                onClick: function () {
                    props.setData({ action: action, structure_id: props.data.structure_id });
                },
                style: {
                    color: color,
                    backgroundColor: "transparent",
                    border: "1px solid " + color,
                    borderRadius: "4px",
                    padding: "2px 10px",
                    marginRight: "6px",
                    fontSize: "12px",
                    cursor: "pointer",
                },
            },
            label
        );
    }

    var buttons = [];
    if (status === "SHELL") {
        buttons.push(button("Enter Trade", "enter_trade", colors.ACCENT_GREEN));
    }
    buttons.push(button("View", "view", colors.TEXT_PRIMARY));
    if (status === "OPEN") {
        buttons.push(button("Exit", "exit", colors.ACCENT_RED));
    }
    return React.createElement("div", { style: { display: "flex", alignItems: "center", height: "100%" } }, buttons);
};

// Generic single-button cell: cellRendererParams {label, action, colors}. Clicking reports
// {action, structure_id} through cellRendererData, like the other renderers above.
dagcomponentfuncs.ButtonRenderer = function (props) {
    var colors = props.colors || {};
    var color = colors.ACCENT_GREEN;
    return React.createElement(
        "button",
        {
            onClick: function () {
                props.setData({ action: props.action, structure_id: props.data.structure_id });
            },
            style: {
                color: color,
                backgroundColor: "transparent",
                border: "1px solid " + color,
                borderRadius: "4px",
                padding: "2px 10px",
                fontSize: "12px",
                cursor: "pointer",
            },
        },
        props.label
    );
};
