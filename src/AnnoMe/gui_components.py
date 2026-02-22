"""Shared GUI components used by FilterGUI and ClassificationGUI."""

import pathlib

from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QPushButton,
    QStyle,
    QStylePainter,
    QTabBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QFontMetrics, QIcon, QPainter, QPixmap


class RotatedTabBar(QTabBar):
    """A QTabBar that draws tab text horizontally for West-positioned tabs (sidebar style).

    Supports:
      - Optional QIcon per tab (drawn to the left of the text)
      - Collapsed mode: shows only the short label (e.g. "1.") and icon
      - Minimal tab height that wraps tightly around the text
    """

    _ICON_SIZE = 18  # px – icon square dimension
    _ICON_TEXT_GAP = 6  # px – spacing between icon and text
    _H_PADDING = 10  # px – left / right padding inside tab
    _V_PADDING = 6  # px – top / bottom padding
    _COLLAPSED_WIDTH = 48  # px – sidebar width when collapsed (icon + short label)
    _EXPANDED_WIDTH = 210  # px – sidebar width when expanded
    _ICON_ONLY_WIDTH = 38  # px – sidebar width when icon-only mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self._collapsed = False
        self._icon_only = False
        # Parallel list of full tab names; short names are derived as "1.", "2." …
        self._full_labels: list[str] = []
        self._short_labels: list[str] = []

    # -- public API ----------------------------------------------------------

    def addTabLabel(self, full_label: str, short_label: str | None = None):
        """Register full & short labels for the next tab."""
        self._full_labels.append(full_label)
        self._short_labels.append(short_label or f"{len(self._full_labels)}.")

    def _refresh_layout(self):
        """Force the tab bar and parent QTabWidget to re-layout after size changes."""
        self.update()
        self.updateGeometry()
        # Set a fixed width on the tab bar so QTabWidget allocates the right space
        w = self.tabSizeHint(0).width() if self.count() > 0 else self._EXPANDED_WIDTH
        self.setFixedWidth(w)
        if self.parent():
            self.parent().updateGeometry()
            self.parent().update()
            # Process events so the layout recalculates immediately
            QApplication.processEvents()

    def setCollapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self._refresh_layout()

    def isCollapsed(self) -> bool:
        return self._collapsed

    def setIconOnly(self, icon_only: bool):
        """Show only icons (no text at all). Triggered by double-click."""
        self._icon_only = icon_only
        self._refresh_layout()

    def isIconOnly(self) -> bool:
        return self._icon_only

    def mouseDoubleClickEvent(self, event):
        """Toggle icon-only mode on double-click."""
        self.setIconOnly(not self._icon_only)
        # If entering icon-only, also clear collapsed so they don't conflict
        if self._icon_only:
            self._collapsed = False
        event.accept()

    # -- size ----------------------------------------------------------------

    def tabSizeHint(self, index):
        fm = QFontMetrics(self.font())
        text_height = fm.height()
        h = text_height + 2 * self._V_PADDING

        if self._icon_only:
            w = self._ICON_ONLY_WIDTH
        elif self._collapsed:
            w = self._COLLAPSED_WIDTH
        else:
            w = self._EXPANDED_WIDTH

        return QSize(w, h)

    # -- paint ---------------------------------------------------------------

    def paintEvent(self, event):
        painter = QStylePainter(self)
        from PyQt5.QtWidgets import QStyleOptionTab

        for i in range(self.count()):
            opt = QStyleOptionTab()
            self.initStyleOption(opt, i)
            tab_rect = opt.rect

            # Let Qt draw tab background (no text / no icon)
            opt.text = ""
            opt.icon = QIcon()
            painter.drawControl(QStyle.CE_TabBarTab, opt)

            painter.save()

            x = tab_rect.left() + self._H_PADDING
            cy = tab_rect.center().y()

            # Draw icon if present
            icon = self.tabIcon(i)
            if not icon.isNull():
                # Centre icon when icon-only
                if self._icon_only:
                    icon_x = tab_rect.left() + (tab_rect.width() - self._ICON_SIZE) // 2
                else:
                    icon_x = x
                icon_y = cy - self._ICON_SIZE // 2
                icon.paint(painter, icon_x, icon_y, self._ICON_SIZE, self._ICON_SIZE)
                if not self._icon_only:
                    x += self._ICON_SIZE + self._ICON_TEXT_GAP

            # Skip text in icon-only mode
            if not self._icon_only:
                # Choose label
                if self._collapsed:
                    label = self._short_labels[i] if i < len(self._short_labels) else f"{i + 1}."
                else:
                    label = self._full_labels[i] if i < len(self._full_labels) else self.tabText(i)

                text_rect = QRect(x, tab_rect.top(), tab_rect.right() - x - 4, tab_rect.height())
                painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, label)

            painter.restore()


class CollapsibleHelpPanel(QWidget):
    """A help panel that collapses on double-click and expands on single-click.

    When expanded, shows the full QGroupBox with help text.
    When collapsed, shows only a small help icon button (❓).
    Double-click the help group box to collapse it.
    Single-click the ❓ icon button to expand it again.
    """

    _ICON_BTN_SIZE = 36  # px – size of the icon-only button

    def __init__(self, help_html: str, max_width: int = 350, parent=None):
        super().__init__(parent)
        self._expanded = True

        # --- Expanded view: full help group box ---
        self._help_group = QGroupBox("Help")
        help_layout = QVBoxLayout()
        help_text = QTextEdit()
        help_text.setReadOnly(True)
        help_text.setHtml(help_html)
        help_layout.addWidget(help_text)
        self._help_group.setLayout(help_layout)
        self._help_group.setMaximumWidth(max_width)

        # --- Collapsed view: icon-only button ---
        self._icon_btn = QPushButton("\u2753")  # ❓
        self._icon_btn.setToolTip("Click to expand help")
        self._icon_btn.setFixedSize(self._ICON_BTN_SIZE, self._ICON_BTN_SIZE)
        self._icon_btn.setStyleSheet("font-size: 18px; padding: 0px;")
        self._icon_btn.hide()
        self._icon_btn.clicked.connect(self._expand)

        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._icon_btn, 0, Qt.AlignTop | Qt.AlignHCenter)
        layout.addWidget(self._help_group, 1)
        self.setLayout(layout)

        # Install event filter on help group to detect double-click → collapse
        self._help_group.installEventFilter(self)

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent

        if obj is self._help_group and event.type() == QEvent.MouseButtonDblClick:
            self._collapse()
            return True
        return super().eventFilter(obj, event)

    def _collapse(self):
        if self._expanded:
            self._expanded = False
            self._help_group.hide()
            self._icon_btn.show()
            self.setMaximumWidth(self._ICON_BTN_SIZE + 8)

    def _expand(self):
        if not self._expanded:
            self._expanded = True
            self._icon_btn.hide()
            self._help_group.show()
            self.setMaximumWidth(16777215)  # QWIDGETSIZE_MAX


def make_text_icon(char: str, size: int = 64, color: str = "#88A0A0") -> QIcon:
    """Create a simple single-character icon rendered into a QPixmap."""
    from PyQt5.QtGui import QFont, QColor

    pix = QPixmap(size, size)
    pix.fill(QColor(0, 0, 0, 0))  # transparent
    p = QPainter(pix)
    p.setPen(QColor(color))
    font = QFont("Segoe UI", int(size * 0.5), QFont.Bold)
    p.setFont(font)
    p.drawText(QRect(0, 0, size, size), Qt.AlignCenter, char)
    p.end()
    return QIcon(pix)


def load_stylesheet():
    """Load the MotionDesk-inspired dark stylesheet."""
    qss_path = pathlib.Path(__file__).parent / "style_motiondesk.qss"
    if qss_path.exists():
        return qss_path.read_text(encoding="utf-8")
    return ""
