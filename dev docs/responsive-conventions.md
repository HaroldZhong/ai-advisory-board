# Responsive Conventions

The desktop app supports a minimum window size of `960x600`. Layouts below
that width are unsupported and may break because this is not a phone-first
interface.

## Breakpoints

- Default styles target the minimum desktop width (`960px` to `1023px`).
- `md:` is reserved for compact desktop refinements (`768px` to `1023px`) when
  a component needs compatibility with embedded WebView behavior.
- `lg:` (`1024px+`) is the full desktop target: expanded sidebar, full labels,
  and inline secondary controls.
- Do not add `sm:` or phone-specific layouts unless the product strategy changes.

## Sidebar

- Below `lg`, the sidebar should collapse to an icon-only rail.
- At `lg` and above, the sidebar may be expanded or user-collapsed.
- Collapsed navigation items need accessible labels or tooltips because text
  labels are hidden.

## Modals

- Dialogs should fit within `90vw` and cap at their intended desktop width.
- Use patterns like `max-w-[min(90vw,640px)]`.
- Long dialog content should scroll internally rather than pushing actions below
  the viewport.
