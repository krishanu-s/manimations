import { createRoot } from "react-dom/client";
import DistInterpolation from "./DistInterpolation";

const root = document.getElementById("dist-interpolation-root");
if (root) createRoot(root).render(<DistInterpolation />);
