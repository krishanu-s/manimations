import { createRoot } from "react-dom/client";
import FellerDiffusion from "./FellerDiffusion";

const root = document.getElementById("feller-root");
if (root) createRoot(root).render(<FellerDiffusion />);
