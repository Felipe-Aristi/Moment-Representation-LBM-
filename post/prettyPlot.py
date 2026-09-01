# prettyplot.py
import math
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams.update({
    "text.usetex": False
})

# mpl.rcParams.update({
#     "text.usetex": True,                # ativa LaTeX externo
#     "font.family": "serif",             # usa família serif
#     "font.serif": ["Computer Modern"],  # fonte padrão do LaTeX
#     "text.latex.preamble": r"\usepackage{amsmath}",
# })
# ----------------------------- utilidades -------------------------------- #

def _to_inches(points):
    return float(points) / 72.0

def _linspace_ticks(a, b, n):
    # garante ao menos 2 ticks quando n<2
    n = max(int(n), 2)
    return np.linspace(float(a), float(b), n)

def _format_fraction(x, max_den=16):
    # aproxima fração com denominador pequeno (estilo simples)
    frac = Fraction(x).limit_denominator(max_den)
    # usa mathtext
    if frac.denominator == 1:
        return f"{frac.numerator}"
    return rf"\frac{{{frac.numerator}}}{{{frac.denominator}}}"

def _compute_sci_exponent(a, b):
    # pega ordem de grandeza "dominante" da faixa (evita 0)
    span = max(abs(a), abs(b), 1e-12)
    exp = int(math.floor(math.log10(span)))
    # evite expoente 0; se 0 não precisamos mostrar
    return exp if exp != 0 else None

def _create_ticks_and_labels(lmin, lmax, n_ticks, fmt, latex, fractions, sci_notation):
    ticks = _linspace_ticks(lmin, lmax, n_ticks)
    exponent_label = ""
    labels = []

    def format_zero(t):
        if np.isclose(t, 0.0, atol=1e-14):
            return r"$0$" if latex else "0"
        return None

    if fractions:
        for t in ticks:
            z = format_zero(t)
            if z is not None:
                labels.append(z)
            else:
                labels.append(rf"${_format_fraction(t)}$")

    elif sci_notation:
        exp = _compute_sci_exponent(lmin, lmax)

        if exp is None:
            # cai para formato normal
            for t in ticks:
                z = format_zero(t)
                if z is not None:
                    labels.append(z)
                else:
                    if fmt == 5:
                        lab = f"{t:g}"
                    else:
                        lab = f"{t:.{int(fmt)}f}"
                    labels.append(rf"${lab}$" if latex else lab)

        else:
            scale = 10.0 ** exp
            base_vals = ticks / scale
            exponent_label = rf"$\times 10^{{{exp}}}$" if latex else f"x 10^{exp}"

            for t, v in zip(ticks, base_vals):
                z = format_zero(t)
                if z is not None:
                    labels.append(z)
                else:
                    if fmt == 5:
                        lab = f"{v:g}"
                    else:
                        lab = f"{v:.{int(fmt)}f}"
                    labels.append(rf"${lab}$" if latex else lab)

    else:
        for t in ticks:
            z = format_zero(t)
            if z is not None:
                labels.append(z)
            else:
                if fmt == 5:
                    lab = f"{t:g}"
                else:
                    lab = f"{t:.{int(fmt)}f}"
                labels.append(rf"${lab}$" if latex else lab)

    return ticks, labels, exponent_label


def _add_exponent_text(ax_like, exponent, where, fontsize, fontname, latex, offset_pts=(0,0)):
    """Desenha o rótulo de expoente '×10^{e}' próximo ao eixo/colorbar.
       where: 'x' | 'y' | 'cbar-vertical' | 'cbar-horizontal'
       offset_pts: deslocamento (x,y) em points, convertido via axes.transAxes + offset_transform
    """
    if not exponent:
        return
    # converte offset em fraction do eixo aproximando 1 eixo ~ 72 pts (boa aproximação visual)
    # opcionalmente poderia medir bbox, mas manteremos leve.
    ox = offset_pts[0] / 72.0
    oy = offset_pts[1] / 72.0

    if where == "x":
        # canto direito-abaixo do eixo
        ax_like.text(0.965 + ox, -0.075 + oy, exponent,
                     transform=ax_like.transAxes, ha="left", va="top",
                     fontsize=fontsize, fontname=fontname)
    elif where == "y":
        # canto esquerdo-acima do eixo
        ax_like.text(0.005 + ox, 1.02 + oy, exponent,
                     transform=ax_like.transAxes, ha="left", va="bottom",
                     fontsize=fontsize, fontname=fontname)
    elif where == "cbar-vertical":
        # topo à direita da barra
        ax_like.text(1.05 + ox, 1.02 + oy, exponent,
                     transform=ax_like.transAxes, ha="left", va="bottom",
                     fontsize=fontsize, fontname=fontname)
    elif where == "cbar-horizontal":
        # extremo direito abaixo da barra
        ax_like.text(1.02 + ox, -0.30 + oy, exponent,
                     transform=ax_like.transAxes, ha="left", va="top",
                     fontsize=fontsize, fontname=fontname)

def _colorbar_position_args(cbar_location, cbar_width_scale, paper_points, pad_pts):
    """Converte a API MATLAB para os kwargs do matplotlib.colorbar:
       - fraction: fração da largura/altura do eixo
       - pad: distância entre eixo e colorbar em fraction da largura/altura do eixo
    """
    # fraction ~ largura da cbar relativa ao eixo
    # cBarWidthScale é fração da largura do "papel"; convertamos heurísticamente
    # para fração do eixo (0.01 em papel ≈ 0.06 do eixo por experiência)
    fraction = float(cbar_width_scale) * 6.0
    fraction = max(0.02, min(fraction, 0.3))  # limites razoáveis

    # pad em points do MATLAB -> transformar para fração (aprox 1 frac ~ 72 pts)
    pad_frac = float(pad_pts) / 72.0
    pad_frac = max(0.001, min(pad_frac, 0.01))

    loc_map = {
        'East':   'right',
        'West':   'left',
        'North':  'top',
        'South':  'bottom',
        'right':  'right',
        'left':   'left',
        'top':    'top',
        'bottom': 'bottom'
    }
    loc = loc_map.get(str(cbar_location), 'right')
    return dict(location=loc, fraction=fraction, pad=pad_frac)


def _ensure_cmap_object(cmap):
    """
    Aceita string (nome) ou um mpl.colors.Colormap.
    Retorna sempre um Colormap válido.
    Não altera comportamento antigo, só evita casos tipo name='from_list'.
    """
    # caso comum: string com nome do cmap
    if isinstance(cmap, str):
        # compat: alguns usuários passam "balance" mas o registrado é "cmo.balance"
        if cmap == "balance":
            cmap = "cmo.balance"
        return plt.get_cmap(cmap)

    # caso comum: já é um Colormap
    if isinstance(cmap, mpl.colors.Colormap):
        return cmap

    # fallback: tenta usar atributo .name (mas evita "from_list")
    name = getattr(cmap, "name", None)
    if isinstance(name, str) and name and name != "from_list":
        try:
            return plt.get_cmap(name)
        except Exception:
            pass

    # último fallback seguro
    return plt.get_cmap("viridis")

# --------------------------- função principal ----------------------------- #

def pretty_plot(
    # Axis
    xLim=(0, 1), yLim=(0, 1), cLim=(0, 1),
    nxTicks=11, nyTicks=11, ncBarTicks=5,
    usexAxis=True, useyAxis=True, useColorBar=False,
    useMinorTick=False, useGrid=False, useBorder=True,
    dataAspectRatio=None,                  # None | (dx,dy,dz) -> usa 'equal'
    plotAspectRatio=(1, 1, 1),

    # Labels
    xLabel="", yLabel="", cBarLabel="", title="",
    boxMarginScale=0.09,
    axLabelOffset=0.75, cBarLabelOffset=0.5,
    cBarWidthScale=0.00785,
    xTickFormat=5, yTickFormat=5, cBarTickFormat=5,
    xLabelAngle=0, yLabelAngle=90, cBarLabelAngle = 0,
    xScientificNotation=False, yScientificNotation=False, cBarScientificNotation=False,
    xTickFractions=False, yTickFractions=False, cBarTickFractions=False,
    xTickLabels=None, yTickLabels=None, cBarTickLabels=None,

    # Colorbar
    cBarLocation='East', cBarTickLabelAngle=None, cBarTickSide="auto",   # "auto" | "top" | "bottom"

    # Font
    fontName='DejaVu Sans', fontUnits='points', fontSize=10,
    labelInterpreter="latex",   # compat: usamos mathtext; não força usetex.
    
    # Paper / layout (em points, como no MATLAB)
    textWidth=1.0, paperPoints=595, marginPoints=90,

    # Figura
    dpi=300, cmap="viridis"

):
    """
    Tradução fiel do prettyPlot.m para Python/Matplotlib.
    Retorna (fig, ax, cbar).
    
    """
    # ----- garantir cmap como objeto Colormap (evita erros tipo 'from_list') -----
    cmap_obj = _ensure_cmap_object(cmap)

    # ----- cores / linewidth (emulate MATLAB) -----
    white = (1.0, 1.0, 1.0)
    black = (0.0, 0.0, 0.0)
    lineWidth = 0.5  # ~ 1/2 pt

    # ----- aspect box ratio como no MATLAB -----
    if dataAspectRatio is not None and len(dataAspectRatio) >= 2:
        boxAspectRatio = (xLim[1] - xLim[0]) / (yLim[1] - yLim[0])
        use_data_aspect = True
    else:
        boxAspectRatio = float(plotAspectRatio[0]) / float(plotAspectRatio[1])
        use_data_aspect = False

    # ----- tamanhos em points -> layout do "papel" -----
    xFigWidth_pts = (paperPoints - 2*marginPoints) * textWidth
    boxMarginWidth_pts = (boxMarginScale * xFigWidth_pts) / textWidth
    xAxisWidth_pts = xFigWidth_pts - 2*boxMarginWidth_pts
    yAxisWidth_pts = xAxisWidth_pts / boxAspectRatio
    yFigWidth_pts = yAxisWidth_pts + 2*boxMarginWidth_pts

    # posição dos rótulos em points (vamos traduzir p/ coordenadas do axes mais tarde)
    titlePosition_pts   = (xAxisWidth_pts/2.0, yAxisWidth_pts + (boxMarginWidth_pts/2.0), 0.0)
    xLabelPosition_pts  = (xAxisWidth_pts/2.0, -boxMarginWidth_pts * axLabelOffset, 0.0)
    yLabelPosition_pts  = (-boxMarginWidth_pts * axLabelOffset, yAxisWidth_pts/2.0, 2.0)

    xExponentPosition_pts   = (xAxisWidth_pts - 2.0*fontSize, 0.0, 0.0)  # usamos opção alternativa com _add_exponent_text
    yExponentPosition_pts   = (fontSize/2.0, yAxisWidth_pts + fontSize, 0.0)

    # ----- figura e eixo -----
    fig_w_in = _to_inches(xFigWidth_pts)
    fig_h_in = _to_inches(yFigWidth_pts)
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor(white)

    # posição do axes em coordinates de figura: converter points -> fraction
    left   = _to_inches(boxMarginWidth_pts) / fig_w_in
    bottom = _to_inches(boxMarginWidth_pts) / fig_h_in
    ax_w   = _to_inches(xAxisWidth_pts) / fig_w_in
    ax_h   = _to_inches(yAxisWidth_pts) / fig_h_in
    ax.set_position([left, bottom, ax_w, ax_h])

    # ----- estilo do eixo -----
    for spine in ax.spines.values():
        spine.set_visible(True if useBorder else False)
        spine.set_linewidth(lineWidth)
        spine.set_color(black)

    ax.set_facecolor(white)
    ax.tick_params(colors=black)
    if useGrid == True:
        ax.grid(useGrid, color=black, linewidth=0.4)
    else:
        ax.grid(useGrid)
    if useMinorTick:
        ax.minorticks_on()
    else:
        ax.minorticks_off()

    # ----- limites -----
    ax.set_xlim(xLim)
    ax.set_ylim(yLim)

    # ----- proporção -----
    if use_data_aspect:
        ax.set_aspect('equal', adjustable='box')  # dataAspectRatio 2D
    else:
        ax.set_box_aspect(float(yAxisWidth_pts)/float(xAxisWidth_pts))

    # ----- fontes -----
    ax.tick_params(labelsize=fontSize)
    # (Matplotlib usa mathtext para $...$ por padrão; não ativamos usetex global.)

    # ----- ticks e labels (X) -----
    if xTickLabels is None or len(xTickLabels) == 0:
        x_ticks, x_ticklabels, x_exp = _create_ticks_and_labels(
            xLim[0], xLim[1], nxTicks, xTickFormat,
            latex=(labelInterpreter == "latex"),
            fractions=xTickFractions,
            sci_notation=xScientificNotation
        )
    else:
        x_ticks = _linspace_ticks(xLim[0], xLim[1], len(xTickLabels))
        x_ticklabels = list(xTickLabels)
        x_exp = ""
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.set_xticklabels(x_ticklabels, rotation=0)

    # visibilidade do eixo X
    ax.get_xaxis().set_visible(bool(usexAxis))

    # ----- ticks e labels (Y) -----
    if yTickLabels is None or len(yTickLabels) == 0:
        y_ticks, y_ticklabels, y_exp = _create_ticks_and_labels(
            yLim[0], yLim[1], nyTicks, yTickFormat,
            latex=(labelInterpreter == "latex"),
            fractions=yTickFractions,
            sci_notation=yScientificNotation
        )
    else:
        y_ticks = _linspace_ticks(yLim[0], yLim[1], len(yTickLabels))
        y_ticklabels = list(yTickLabels)
        y_exp = ""
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.set_yticklabels(y_ticklabels, rotation=0)

    # visibilidade do eixo Y
    ax.get_yaxis().set_visible(bool(useyAxis))

    # ----- títulos / rótulos (com rotação compatível) -----
    # Em matplotlib, colocar "em points" requer conversion; adotamos offset típico e aplicamos rotação.
    ax.set_title(title, fontsize=fontSize, fontname=fontName, pad=6.0)
    ax.set_xlabel(xLabel, fontsize=fontSize, fontname=fontName, labelpad=6.0, rotation=float(xLabelAngle))
    ax.set_ylabel(yLabel, fontsize=fontSize, fontname=fontName, labelpad=6.0, rotation=float(yLabelAngle))
    
    # ----- títulos / rótulos -----
    ax.set_title(title, fontsize=fontSize, fontname=fontName, pad=6.0)

    ax.set_xlabel(xLabel, fontsize=fontSize, fontname=fontName, rotation=float(xLabelAngle))
    ax.set_ylabel(yLabel, fontsize=fontSize, fontname=fontName, rotation=float(yLabelAngle))

    # ====== FIX: coordenadas exatas dos labels (em fração do Axes) ======
    xlab_x = 0.5
    xlab_y = (-boxMarginWidth_pts * axLabelOffset) / yAxisWidth_pts   # negativo -> abaixo
    ax.xaxis.set_label_coords(xlab_x, xlab_y)

    ylab_x = (-boxMarginWidth_pts * axLabelOffset) / xAxisWidth_pts   # negativo -> à esquerda
    ylab_y = 0.5
    ax.yaxis.set_label_coords(ylab_x, ylab_y)

    # ====== FIX: alinhamento/âncora (muito importante quando yLabelAngle=0) ======
    ax.xaxis.label.set_ha("center")
    ax.xaxis.label.set_va("top")

    ax.yaxis.label.set_ha("right")   # fica “encostando” no eixo quando horizontal
    ax.yaxis.label.set_va("center")
    ax.yaxis.label.set_rotation_mode("anchor")
    ax.yaxis.set_label_coords(-0.1, 0.5)  # x>1 empurra para a direita; y=0.5 centraliza


    # ----- colorbar -----
    cbar = None
    # plt.set_cmap(cmap)
    cbar_exp = ""
    
    
    if useColorBar:
        # Colormap padrão do eixo (dispensa passar cmap no contourf)
        # plt.set_cmap(cmap)


        # Mapeador para a colorbar (independente do contourf)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cLim[0], vmax=cLim[1]))
        sm.set_array([])

        # ===== medir tamanho do Axes em points (para espessura/pad "inteligentes") =====
        fig.canvas.draw()  # garante bbox atualizado
        bbox_in = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        ax_w_pts = float(bbox_in.width * 72.0)
        ax_h_pts = float(bbox_in.height * 72.0)

        loc = str(cBarLocation).lower()

        # ===== escolher espessura/pad com limites (points) =====
        if loc in ("east", "west", "right", "left"):
            # vertical: espessura depende da largura do Axes
            cb_thickness_pts = float(np.clip(0.03 * ax_w_pts, 2.5, 5.0))
            cb_pad_pts       = float(np.clip(0.02 * ax_w_pts,  4.0, 12.0))
        else:
            # horizontal: espessura depende da altura do Axes
            cb_thickness_pts = float(np.clip(0.03 * ax_h_pts, 10.0, 22.0))
            cb_pad_pts       = float(np.clip(0.02 * ax_h_pts,  4.0, 12.0))

        # ===== criar cax manualmente (garante MESMO tamanho do plot) =====
        pos = ax.get_position()  # Bbox em fração da figura (0-1)

        # points -> inches
        cb_thickness_in = cb_thickness_pts / 72.0
        cb_pad_in       = cb_pad_pts       / 72.0

        # inches -> fração da figura
        cb_thick_fx = cb_thickness_in / fig_w_in
        cb_pad_fx   = cb_pad_in       / fig_w_in
        cb_thick_fy = cb_thickness_in / fig_h_in
        cb_pad_fy   = cb_pad_in       / fig_h_in

        if loc in ("east", "right"):
            cax = fig.add_axes([pos.x1 + cb_pad_fx, pos.y0, cb_thick_fx, pos.height])
            orientation = "vertical"
        elif loc in ("west", "left"):
            cax = fig.add_axes([pos.x0 - cb_pad_fx - cb_thick_fx, pos.y0, cb_thick_fx, pos.height])
            orientation = "vertical"
        elif loc in ("north", "top"):
            cax = fig.add_axes([pos.x0, pos.y1 + cb_pad_fy, pos.width, cb_thick_fy])
            orientation = "horizontal"
        else:  # ("south","bottom")
            cax = fig.add_axes([pos.x0, pos.y0 - cb_pad_fy - cb_thick_fy, pos.width, cb_thick_fy])
            orientation = "horizontal"

        # ===== ticks / labels =====
        if cBarTickLabels is None or len(cBarTickLabels) == 0:
            ticks_for_cbar, labels_for_cbar, cbar_exp = _create_ticks_and_labels(
                cLim[0], cLim[1], ncBarTicks, cBarTickFormat,
                latex=(labelInterpreter == "latex"),
                fractions=cBarTickFractions,
                sci_notation=cBarScientificNotation
            )
        else:
            ticks_for_cbar = _linspace_ticks(cLim[0], cLim[1], len(cBarTickLabels))
            labels_for_cbar = list(cBarTickLabels)
            cbar_exp = ""

        # ===== criar colorbar =====
        cbar = fig.colorbar(sm, cax=cax, ticks=ticks_for_cbar, orientation=orientation)
        cbar.set_ticklabels(labels_for_cbar)
        cbar.ax.tick_params(labelsize=fontSize)

        # label da colorbar
        cbar.set_label(
            cBarLabel,
            fontsize=fontSize,
            fontname=fontName,
            rotation=float(cBarLabelAngle),
        )
        
        if orientation == "horizontal":
            side = cBarTickSide
            if side == "auto":
                side = "top" if loc in ("north", "top") else "bottom"

            if side == "top":
                cbar.ax.xaxis.set_ticks_position("top")
                cbar.ax.tick_params(axis="x", labeltop=True, labelbottom=False)
            else:
                cbar.ax.xaxis.set_ticks_position("bottom")
                cbar.ax.tick_params(axis="x", labeltop=False, labelbottom=True)


        # centralizar label de forma robusta
        if orientation == "vertical":
            # x controla distância do texto; y=0.5 garante centro
            cbar.ax.yaxis.set_label_coords(5, 0.5)
            cbar.ax.yaxis.label.set_va("center")
            cbar.ax.yaxis.label.set_ha("left")
            cbar.ax.yaxis.label.set_rotation_mode("anchor")
        else:
            cbar.ax.xaxis.set_label_coords(0.5, -1.2)
            cbar.ax.xaxis.label.set_ha("center")
            cbar.ax.xaxis.label.set_va("top")
            cbar.ax.xaxis.label.set_rotation_mode("anchor")
            
        if orientation == "horizontal" and loc in ("north", "top"):
            cbar.ax.xaxis.set_label_coords(0.5, 3.5)  # ajuste fino: 1.6–2.4
            cbar.ax.xaxis.label.set_va("bottom")

        # rotação dos ticks da cbar (se pedido)
        if cBarTickLabelAngle is not None:
            if orientation == "vertical":
                for t in cbar.ax.get_yticklabels():
                    t.set_rotation(float(cBarTickLabelAngle))
            else:
                for t in cbar.ax.get_xticklabels():
                    t.set_rotation(float(cBarTickLabelAngle))

        # expoente da cbar (se aplicável)
        if cBarScientificNotation and cbar is not None and cbar_exp:
            where = "cbar-vertical" if orientation == "vertical" else "cbar-horizontal"
            _add_exponent_text(
                cbar.ax, cbar_exp, where,
                fontsize=fontSize, fontname=fontName,
                latex=(labelInterpreter == "latex"),
                offset_pts=(0, 0)
            )



    # if useColorBar:
    #     # criamos um mapeador "vazio" só para a colorbar (como no MATLAB quando ainda não há imagem)
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cLim[0], vmax=cLim[1]))
    #     sm.set_array([])

    #     # converter posição/espessura no estilo MATLAB
    #     pad_pts = max(2.5, 0.0025 * float(paperPoints))  # mesma ideia do código MATLAB comentado
    #     cbar_kwargs = _colorbar_position_args(cBarLocation, cBarWidthScale, paperPoints, pad_pts)

    #     # ticks / labels da cbar
    #     if cBarTickLabels is None or len(cBarTickLabels) == 0:
    #         cbar_ticks = _linspace_ticks(cLim[0], cLim[1], ncBarTicks)
    #         cbar_ticks2, cbar_ticklabels, cbar_exp = _create_ticks_and_labels(
    #             cLim[0], cLim[1], ncBarTicks, cBarTickFormat,
    #             latex=(labelInterpreter == "latex"),
    #             fractions=cBarTickFractions,
    #             sci_notation=cBarScientificNotation
    #         )
    #         # garantimos alinhar ticks numéricos com rótulos formatados
    #         ticks_for_cbar = cbar_ticks
    #         labels_for_cbar = cbar_ticklabels
    #     else:
    #         ticks_for_cbar = _linspace_ticks(cLim[0], cLim[1], len(cBarTickLabels))
    #         labels_for_cbar = list(cBarTickLabels)

    #     cbar = fig.colorbar(sm, ax=ax, ticks=ticks_for_cbar, **cbar_kwargs)
    #     cbar.set_ticklabels(labels_for_cbar)
    #     cbar.set_label(cBarLabel, fontsize=fontSize, fontname=fontName, labelpad=6.0, rotation=float(cBarLabelAngle))
    #     cbar.ax.tick_params(labelsize=fontSize)
        
    #     # depois de: cbar.set_label(...)
    #     cbar.ax.yaxis.set_label_coords(2.5, 0.5)  # x>1 empurra para a direita; y=0.5 centraliza

    #     cbar.ax.yaxis.label.set_ha("left")
    #     cbar.ax.yaxis.label.set_va("center")
    #     cbar.ax.yaxis.label.set_rotation_mode("anchor")

        

    #     # rotação dos ticks da cbar (se pedido)
    #     if cBarTickLabelAngle is not None:
    #         if cbar_kwargs['location'] in ('right', 'left'):
    #             for t in cbar.ax.get_yticklabels():
    #                 t.set_rotation(float(cBarTickLabelAngle))
    #         else:
    #             for t in cbar.ax.get_xticklabels():
    #                 t.set_rotation(float(cBarTickLabelAngle))

    #     # expoente da cbar
    #     if cBarScientificNotation and cbar is not None and cbar_exp:
    #         where = "cbar-vertical" if cbar_kwargs['location'] in ('right', 'left') else "cbar-horizontal"
    #         _add_exponent_text(cbar.ax, cbar_exp, where,
    #                            fontsize=fontSize, fontname=fontName,
    #                            latex=(labelInterpreter == "latex"),
    #                            offset_pts=(0, 0))

    # ----- borda/moldura "manual" dentro dos limites de dados -----
    if useBorder:
        ax.plot([xLim[0], xLim[1], xLim[1], xLim[0], xLim[0]],
                [yLim[0], yLim[0], yLim[1], yLim[1], yLim[0]],
                color=black, linewidth=lineWidth, solid_capstyle='butt',
                zorder=10, clip_on=False)

    # ----- expoentes (x/y) -----
    if xScientificNotation and x_exp:
        _add_exponent_text(ax, x_exp, where="x",
                           fontsize=fontSize, fontname=fontName,
                           latex=(labelInterpreter == "latex"),
                           offset_pts=(0, 0))
    if yScientificNotation and y_exp:
        _add_exponent_text(ax, y_exp, where="y",
                           fontsize=fontSize, fontname=fontName,
                           latex=(labelInterpreter == "latex"),
                           offset_pts=(0, 0))

    return fig, ax, cbar