import { createRoot } from "react-dom/client";
import CLTInterpolation from "./CLTInterpolation";

const root = document.getElementById("clt-root");
if (root) createRoot(root).render(<CLTInterpolation />);
