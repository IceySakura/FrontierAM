import { consolePath, navigateConsole, safeReturnPath } from "./routes.js";

export default function ConsoleLink({
  to,
  origin = true,
  preserveScroll = false,
  children,
  onClick,
  ...props
}) {
  let route = to;
  if (typeof route !== "string" && origin && !route.query?.return_to) {
    const current = safeReturnPath(
      window.location.pathname + window.location.search,
    );
    route = {
      ...route,
      query: { ...route.query, ...(current ? { return_to: current } : {}) },
    };
  }
  const href = typeof route === "string" ? route : consolePath(route);
  return (
    <a
      {...props}
      href={href}
      onClick={(event) => {
        onClick?.(event);
        if (
          event.defaultPrevented ||
          event.button !== 0 ||
          event.metaKey ||
          event.ctrlKey ||
          event.shiftKey ||
          event.altKey ||
          (props.target && props.target !== "_self")
        )
          return;
        event.preventDefault();
        event.stopPropagation();
        navigateConsole(
          route,
          preserveScroll
            ? { state: { scrollY: window.scrollY } }
            : { restore: !origin },
        );
      }}
    >
      {children}
    </a>
  );
}
