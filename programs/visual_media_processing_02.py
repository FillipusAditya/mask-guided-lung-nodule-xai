import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# =========================================================
# FUNCTION : CHECK TRIANGLE-LINE INTERSECTION
# =========================================================

def triangle_line_intersection(A, B, C, O, P):

    # Edge vectors
    AB = B - A
    AC = C - A

    # Normal vector
    N = np.cross(AB, AC)

    # Direction vector of line
    D = P - O

    # Check parallel condition
    denominator = np.dot(N, D)

    if abs(denominator) < 1e-6:
        return False, None

    # Compute parameter t
    t = np.dot(N, A - O) / denominator

    # Check if intersection lies on segment
    if t < 0 or t > 1:
        return False, None

    # Intersection point
    I = O + t * D

    # =========================================
    # Barycentric coordinate test
    # =========================================

    v0 = C - A
    v1 = B - A
    v2 = I - A

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)

    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    inside = (u >= 0) and (v >= 0) and (u + v <= 1)

    if inside:
        return True, I
    else:
        return False, I


# =========================================================
# TRIANGLE DEFINITION
# =========================================================

A = np.array([1, 1, 3])
B = np.array([5, 1, 3])
C = np.array([3, 5, 3])

# Origin
O = np.array([0, 0, 0])

# =========================================================
# CASE 1 : INTERSECTION OCCURS
# =========================================================

P1 = np.array([3, 3, 5])

intersect1, I1 = triangle_line_intersection(A, B, C, O, P1)

# =========================================================
# CASE 2 : NO INTERSECTION
# =========================================================

P2 = np.array([6, 6, 5])

intersect2, I2 = triangle_line_intersection(A, B, C, O, P2)

# =========================================================
# VISUALIZATION
# =========================================================

fig = plt.figure(figsize=(14, 7))

# =========================================================
# FIRST SUBPLOT : INTERSECTION
# =========================================================

ax1 = fig.add_subplot(121, projection='3d')

triangle = [[A, B, C]]

poly1 = Poly3DCollection(
    triangle,
    alpha=0.5,
    facecolor='cyan',
    edgecolor='black'
)

ax1.add_collection3d(poly1)

# Draw line
ax1.plot(
    [O[0], P1[0]],
    [O[1], P1[1]],
    [O[2], P1[2]],
    linewidth=3,
    label='Line Segment'
)

# Draw triangle vertices
ax1.scatter(*A, s=80)
ax1.scatter(*B, s=80)
ax1.scatter(*C, s=80)

# Draw origin and endpoint
ax1.scatter(*O, s=80)
ax1.scatter(*P1, s=80)

# Draw intersection point
if intersect1:
    ax1.scatter(
        I1[0], I1[1], I1[2],
        s=150,
        color='red',
        label='Intersection Point'
    )

# Labels
ax1.set_title("CASE 1 : INTERSECTION OCCURS")

ax1.set_xlim(0, 6)
ax1.set_ylim(0, 6)
ax1.set_zlim(0, 6)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax1.legend()

# =========================================================
# SECOND SUBPLOT : NO INTERSECTION
# =========================================================

ax2 = fig.add_subplot(122, projection='3d')

poly2 = Poly3DCollection(
    triangle,
    alpha=0.5,
    facecolor='orange',
    edgecolor='black'
)

ax2.add_collection3d(poly2)

# Draw line
ax2.plot(
    [O[0], P2[0]],
    [O[1], P2[1]],
    [O[2], P2[2]],
    linewidth=3,
    label='Line Segment'
)

# Draw vertices
ax2.scatter(*A, s=80)
ax2.scatter(*B, s=80)
ax2.scatter(*C, s=80)

# Draw origin and endpoint
ax2.scatter(*O, s=80)
ax2.scatter(*P2, s=80)

# Optional intersection point
if I2 is not None:
    ax2.scatter(
        I2[0], I2[1], I2[2],
        s=120,
        color='purple',
        label='Plane Intersection'
    )

ax2.set_title("CASE 2 : NO INTERSECTION")

ax2.set_xlim(0, 7)
ax2.set_ylim(0, 7)
ax2.set_zlim(0, 6)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

ax2.legend()

# Better viewing angle
ax1.view_init(elev=25, azim=35)
ax2.view_init(elev=25, azim=35)

plt.tight_layout()
plt.show()

# =========================================================
# PRINT RESULTS
# =========================================================

print("========== CASE 1 ==========")

if intersect1:
    print("Intersection occurs.")
    print("Intersection point:", I1)
else:
    print("No intersection.")

print("\n========== CASE 2 ==========")

if intersect2:
    print("Intersection occurs.")
    print("Intersection point:", I2)
else:
    print("No intersection.")