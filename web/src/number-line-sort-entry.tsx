import { createRoot } from "react-dom/client";
import NumberLineSort from "./NumberLineSort";

const root = document.getElementById("number-line-sort-root");
if (root) createRoot(root).render(<NumberLineSort />);
