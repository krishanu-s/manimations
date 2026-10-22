import { createRoot } from "react-dom/client";
import CornerGame from "./CornerGame";

const root = document.getElementById("corner-game-root");
if (root) createRoot(root).render(<CornerGame />);
