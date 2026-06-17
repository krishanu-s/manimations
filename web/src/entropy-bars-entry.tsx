import { createRoot } from "react-dom/client";
import EntropyBars from "./EntropyBars";

const root = document.getElementById("entropy-bars-root");
if (root) createRoot(root).render(<EntropyBars />);
