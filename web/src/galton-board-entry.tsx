import { createRoot } from "react-dom/client";
import GaltonBoard from "./GaltonBoard";

const root = document.getElementById("galton-board-root");
if (root) createRoot(root).render(<GaltonBoard />);
