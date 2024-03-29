async function getGridJs() {
    var __defProp = Object.defineProperty;
    var __export = (target, all) => {
    for (var name in all)
        __defProp(target, name, { get: all[name], enumerable: true });
    };

    // https://cdn.skypack.dev/gridjs@6.0.6
    var gridjs_6_0_exports = {};
    __export(gridjs_6_0_exports, {
    Cell: () => X,
    Component: () => N,
    Config: () => fn,
    Grid: () => In,
    PluginPosition: () => qt,
    Row: () => Z,
    className: () => nt,
    createElement: () => w,
    createRef: () => k,
    default: () => gridjs_default,
    h: () => w,
    html: () => G,
    useConfig: () => Et,
    useEffect: () => gt,
    useRef: () => yt,
    useSelector: () => jt,
    useState: () => vt,
    useStore: () => Ht,
    useTranslator: () => Lt
    });

    // https://cdn.skypack.dev/-/gridjs@v6.0.6-dbZCTh9aYhxYNmUQkdUH/dist=es2019,mode=imports/optimized/gridjs.js
    function t(t2, n2) {
    for (var e2 = 0; e2 < n2.length; e2++) {
        var r2 = n2[e2];
        r2.enumerable = r2.enumerable || false, r2.configurable = true, "value" in r2 && (r2.writable = true), Object.defineProperty(t2, typeof (o2 = function(t3, n3) {
        if (typeof t3 != "object" || t3 === null)
            return t3;
        var e3 = t3[Symbol.toPrimitive];
        if (e3 !== void 0) {
            var r3 = e3.call(t3, "string");
            if (typeof r3 != "object")
            return r3;
            throw new TypeError("@@toPrimitive must return a primitive value.");
        }
        return String(t3);
        }(r2.key)) == "symbol" ? o2 : String(o2), r2);
    }
    var o2;
    }
    function n(n2, e2, r2) {
    return e2 && t(n2.prototype, e2), r2 && t(n2, r2), Object.defineProperty(n2, "prototype", { writable: false }), n2;
    }
    function e() {
    return e = Object.assign ? Object.assign.bind() : function(t2) {
        for (var n2 = 1; n2 < arguments.length; n2++) {
        var e2 = arguments[n2];
        for (var r2 in e2)
            Object.prototype.hasOwnProperty.call(e2, r2) && (t2[r2] = e2[r2]);
        }
        return t2;
    }, e.apply(this, arguments);
    }
    function r(t2, n2) {
    t2.prototype = Object.create(n2.prototype), t2.prototype.constructor = t2, o(t2, n2);
    }
    function o(t2, n2) {
    return o = Object.setPrototypeOf ? Object.setPrototypeOf.bind() : function(t3, n3) {
        return t3.__proto__ = n3, t3;
    }, o(t2, n2);
    }
    function i(t2) {
    if (t2 === void 0)
        throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
    return t2;
    }
    function u(t2, n2) {
    (n2 == null || n2 > t2.length) && (n2 = t2.length);
    for (var e2 = 0, r2 = new Array(n2); e2 < n2; e2++)
        r2[e2] = t2[e2];
    return r2;
    }
    function s(t2, n2) {
    var e2 = typeof Symbol != "undefined" && t2[Symbol.iterator] || t2["@@iterator"];
    if (e2)
        return (e2 = e2.call(t2)).next.bind(e2);
    if (Array.isArray(t2) || (e2 = function(t3, n3) {
        if (t3) {
        if (typeof t3 == "string")
            return u(t3, n3);
        var e3 = Object.prototype.toString.call(t3).slice(8, -1);
        return e3 === "Object" && t3.constructor && (e3 = t3.constructor.name), e3 === "Map" || e3 === "Set" ? Array.from(t3) : e3 === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(e3) ? u(t3, n3) : void 0;
        }
    }(t2)) || n2 && t2 && typeof t2.length == "number") {
        e2 && (t2 = e2);
        var r2 = 0;
        return function() {
        return r2 >= t2.length ? { done: true } : { done: false, value: t2[r2++] };
        };
    }
    throw new TypeError("Invalid attempt to iterate non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
    }
    var a;
    !function(t2) {
    t2[t2.Init = 0] = "Init", t2[t2.Loading = 1] = "Loading", t2[t2.Loaded = 2] = "Loaded", t2[t2.Rendered = 3] = "Rendered", t2[t2.Error = 4] = "Error";
    }(a || (a = {}));
    var l;
    var c;
    var f;
    var p;
    var d;
    var h;
    var _;
    var m = {};
    var v = [];
    var g = /acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i;
    function y(t2, n2) {
    for (var e2 in n2)
        t2[e2] = n2[e2];
    return t2;
    }
    function b(t2) {
    var n2 = t2.parentNode;
    n2 && n2.removeChild(t2);
    }
    function w(t2, n2, e2) {
    var r2, o2, i2, u2 = {};
    for (i2 in n2)
        i2 == "key" ? r2 = n2[i2] : i2 == "ref" ? o2 = n2[i2] : u2[i2] = n2[i2];
    if (arguments.length > 2 && (u2.children = arguments.length > 3 ? l.call(arguments, 2) : e2), typeof t2 == "function" && t2.defaultProps != null)
        for (i2 in t2.defaultProps)
        u2[i2] === void 0 && (u2[i2] = t2.defaultProps[i2]);
    return x(t2, u2, r2, o2, null);
    }
    function x(t2, n2, e2, r2, o2) {
    var i2 = { type: t2, props: n2, key: e2, ref: r2, __k: null, __: null, __b: 0, __e: null, __d: void 0, __c: null, __h: null, constructor: void 0, __v: o2 == null ? ++f : o2 };
    return o2 == null && c.vnode != null && c.vnode(i2), i2;
    }
    function k() {
    return { current: null };
    }
    function S(t2) {
    return t2.children;
    }
    function N(t2, n2) {
    this.props = t2, this.context = n2;
    }
    function C(t2, n2) {
    if (n2 == null)
        return t2.__ ? C(t2.__, t2.__.__k.indexOf(t2) + 1) : null;
    for (var e2; n2 < t2.__k.length; n2++)
        if ((e2 = t2.__k[n2]) != null && e2.__e != null)
        return e2.__e;
    return typeof t2.type == "function" ? C(t2) : null;
    }
    function P(t2) {
    var n2, e2;
    if ((t2 = t2.__) != null && t2.__c != null) {
        for (t2.__e = t2.__c.base = null, n2 = 0; n2 < t2.__k.length; n2++)
        if ((e2 = t2.__k[n2]) != null && e2.__e != null) {
            t2.__e = t2.__c.base = e2.__e;
            break;
        }
        return P(t2);
    }
    }
    function E(t2) {
    (!t2.__d && (t2.__d = true) && d.push(t2) && !I.__r++ || h !== c.debounceRendering) && ((h = c.debounceRendering) || setTimeout)(I);
    }
    function I() {
    for (var t2; I.__r = d.length; )
        t2 = d.sort(function(t3, n2) {
        return t3.__v.__b - n2.__v.__b;
        }), d = [], t2.some(function(t3) {
        var n2, e2, r2, o2, i2, u2;
        t3.__d && (i2 = (o2 = (n2 = t3).__v).__e, (u2 = n2.__P) && (e2 = [], (r2 = y({}, o2)).__v = o2.__v + 1, F(u2, o2, r2, n2.__n, u2.ownerSVGElement !== void 0, o2.__h != null ? [i2] : null, e2, i2 == null ? C(o2) : i2, o2.__h), O(e2, o2), o2.__e != i2 && P(o2)));
        });
    }
    function T(t2, n2, e2, r2, o2, i2, u2, s2, a2, l2) {
    var c2, f2, p2, d2, h2, _2, g2, y2 = r2 && r2.__k || v, b2 = y2.length;
    for (e2.__k = [], c2 = 0; c2 < n2.length; c2++)
        if ((d2 = e2.__k[c2] = (d2 = n2[c2]) == null || typeof d2 == "boolean" ? null : typeof d2 == "string" || typeof d2 == "number" || typeof d2 == "bigint" ? x(null, d2, null, null, d2) : Array.isArray(d2) ? x(S, { children: d2 }, null, null, null) : d2.__b > 0 ? x(d2.type, d2.props, d2.key, d2.ref ? d2.ref : null, d2.__v) : d2) != null) {
        if (d2.__ = e2, d2.__b = e2.__b + 1, (p2 = y2[c2]) === null || p2 && d2.key == p2.key && d2.type === p2.type)
            y2[c2] = void 0;
        else
            for (f2 = 0; f2 < b2; f2++) {
            if ((p2 = y2[f2]) && d2.key == p2.key && d2.type === p2.type) {
                y2[f2] = void 0;
                break;
            }
            p2 = null;
            }
        F(t2, d2, p2 = p2 || m, o2, i2, u2, s2, a2, l2), h2 = d2.__e, (f2 = d2.ref) && p2.ref != f2 && (g2 || (g2 = []), p2.ref && g2.push(p2.ref, null, d2), g2.push(f2, d2.__c || h2, d2)), h2 != null ? (_2 == null && (_2 = h2), typeof d2.type == "function" && d2.__k === p2.__k ? d2.__d = a2 = L(d2, a2, t2) : a2 = A(t2, d2, p2, y2, h2, a2), typeof e2.type == "function" && (e2.__d = a2)) : a2 && p2.__e == a2 && a2.parentNode != t2 && (a2 = C(p2));
        }
    for (e2.__e = _2, c2 = b2; c2--; )
        y2[c2] != null && W(y2[c2], y2[c2]);
    if (g2)
        for (c2 = 0; c2 < g2.length; c2++)
        U(g2[c2], g2[++c2], g2[++c2]);
    }
    function L(t2, n2, e2) {
    for (var r2, o2 = t2.__k, i2 = 0; o2 && i2 < o2.length; i2++)
        (r2 = o2[i2]) && (r2.__ = t2, n2 = typeof r2.type == "function" ? L(r2, n2, e2) : A(e2, r2, r2, o2, r2.__e, n2));
    return n2;
    }
    function A(t2, n2, e2, r2, o2, i2) {
    var u2, s2, a2;
    if (n2.__d !== void 0)
        u2 = n2.__d, n2.__d = void 0;
    else if (e2 == null || o2 != i2 || o2.parentNode == null)
        t:
        if (i2 == null || i2.parentNode !== t2)
            t2.appendChild(o2), u2 = null;
        else {
            for (s2 = i2, a2 = 0; (s2 = s2.nextSibling) && a2 < r2.length; a2 += 1)
            if (s2 == o2)
                break t;
            t2.insertBefore(o2, i2), u2 = i2;
        }
    return u2 !== void 0 ? u2 : o2.nextSibling;
    }
    function H(t2, n2, e2) {
    n2[0] === "-" ? t2.setProperty(n2, e2) : t2[n2] = e2 == null ? "" : typeof e2 != "number" || g.test(n2) ? e2 : e2 + "px";
    }
    function j(t2, n2, e2, r2, o2) {
    var i2;
    t:
        if (n2 === "style")
        if (typeof e2 == "string")
            t2.style.cssText = e2;
        else {
            if (typeof r2 == "string" && (t2.style.cssText = r2 = ""), r2)
            for (n2 in r2)
                e2 && n2 in e2 || H(t2.style, n2, "");
            if (e2)
            for (n2 in e2)
                r2 && e2[n2] === r2[n2] || H(t2.style, n2, e2[n2]);
        }
        else if (n2[0] === "o" && n2[1] === "n")
        i2 = n2 !== (n2 = n2.replace(/Capture$/, "")), n2 = n2.toLowerCase() in t2 ? n2.toLowerCase().slice(2) : n2.slice(2), t2.l || (t2.l = {}), t2.l[n2 + i2] = e2, e2 ? r2 || t2.addEventListener(n2, i2 ? M : D, i2) : t2.removeEventListener(n2, i2 ? M : D, i2);
        else if (n2 !== "dangerouslySetInnerHTML") {
        if (o2)
            n2 = n2.replace(/xlink(H|:h)/, "h").replace(/sName$/, "s");
        else if (n2 !== "href" && n2 !== "list" && n2 !== "form" && n2 !== "tabIndex" && n2 !== "download" && n2 in t2)
            try {
            t2[n2] = e2 == null ? "" : e2;
            break t;
            } catch (t3) {
            }
        typeof e2 == "function" || (e2 == null || e2 === false && n2.indexOf("-") == -1 ? t2.removeAttribute(n2) : t2.setAttribute(n2, e2));
        }
    }
    function D(t2) {
    this.l[t2.type + false](c.event ? c.event(t2) : t2);
    }
    function M(t2) {
    this.l[t2.type + true](c.event ? c.event(t2) : t2);
    }
    function F(t2, n2, e2, r2, o2, i2, u2, s2, a2) {
    var l2, f2, p2, d2, h2, _2, m2, v2, g2, b2, w2, x2, k2, C2, P2, E2 = n2.type;
    if (n2.constructor !== void 0)
        return null;
    e2.__h != null && (a2 = e2.__h, s2 = n2.__e = e2.__e, n2.__h = null, i2 = [s2]), (l2 = c.__b) && l2(n2);
    try {
        t:
        if (typeof E2 == "function") {
            if (v2 = n2.props, g2 = (l2 = E2.contextType) && r2[l2.__c], b2 = l2 ? g2 ? g2.props.value : l2.__ : r2, e2.__c ? m2 = (f2 = n2.__c = e2.__c).__ = f2.__E : ("prototype" in E2 && E2.prototype.render ? n2.__c = f2 = new E2(v2, b2) : (n2.__c = f2 = new N(v2, b2), f2.constructor = E2, f2.render = B), g2 && g2.sub(f2), f2.props = v2, f2.state || (f2.state = {}), f2.context = b2, f2.__n = r2, p2 = f2.__d = true, f2.__h = [], f2._sb = []), f2.__s == null && (f2.__s = f2.state), E2.getDerivedStateFromProps != null && (f2.__s == f2.state && (f2.__s = y({}, f2.__s)), y(f2.__s, E2.getDerivedStateFromProps(v2, f2.__s))), d2 = f2.props, h2 = f2.state, p2)
            E2.getDerivedStateFromProps == null && f2.componentWillMount != null && f2.componentWillMount(), f2.componentDidMount != null && f2.__h.push(f2.componentDidMount);
            else {
            if (E2.getDerivedStateFromProps == null && v2 !== d2 && f2.componentWillReceiveProps != null && f2.componentWillReceiveProps(v2, b2), !f2.__e && f2.shouldComponentUpdate != null && f2.shouldComponentUpdate(v2, f2.__s, b2) === false || n2.__v === e2.__v) {
                for (f2.props = v2, f2.state = f2.__s, n2.__v !== e2.__v && (f2.__d = false), f2.__v = n2, n2.__e = e2.__e, n2.__k = e2.__k, n2.__k.forEach(function(t3) {
                t3 && (t3.__ = n2);
                }), w2 = 0; w2 < f2._sb.length; w2++)
                f2.__h.push(f2._sb[w2]);
                f2._sb = [], f2.__h.length && u2.push(f2);
                break t;
            }
            f2.componentWillUpdate != null && f2.componentWillUpdate(v2, f2.__s, b2), f2.componentDidUpdate != null && f2.__h.push(function() {
                f2.componentDidUpdate(d2, h2, _2);
            });
            }
            if (f2.context = b2, f2.props = v2, f2.__v = n2, f2.__P = t2, x2 = c.__r, k2 = 0, "prototype" in E2 && E2.prototype.render) {
            for (f2.state = f2.__s, f2.__d = false, x2 && x2(n2), l2 = f2.render(f2.props, f2.state, f2.context), C2 = 0; C2 < f2._sb.length; C2++)
                f2.__h.push(f2._sb[C2]);
            f2._sb = [];
            } else
            do {
                f2.__d = false, x2 && x2(n2), l2 = f2.render(f2.props, f2.state, f2.context), f2.state = f2.__s;
            } while (f2.__d && ++k2 < 25);
            f2.state = f2.__s, f2.getChildContext != null && (r2 = y(y({}, r2), f2.getChildContext())), p2 || f2.getSnapshotBeforeUpdate == null || (_2 = f2.getSnapshotBeforeUpdate(d2, h2)), P2 = l2 != null && l2.type === S && l2.key == null ? l2.props.children : l2, T(t2, Array.isArray(P2) ? P2 : [P2], n2, e2, r2, o2, i2, u2, s2, a2), f2.base = n2.__e, n2.__h = null, f2.__h.length && u2.push(f2), m2 && (f2.__E = f2.__ = null), f2.__e = false;
        } else
            i2 == null && n2.__v === e2.__v ? (n2.__k = e2.__k, n2.__e = e2.__e) : n2.__e = R(e2.__e, n2, e2, r2, o2, i2, u2, a2);
        (l2 = c.diffed) && l2(n2);
    } catch (t3) {
        n2.__v = null, (a2 || i2 != null) && (n2.__e = s2, n2.__h = !!a2, i2[i2.indexOf(s2)] = null), c.__e(t3, n2, e2);
    }
    }
    function O(t2, n2) {
    c.__c && c.__c(n2, t2), t2.some(function(n3) {
        try {
        t2 = n3.__h, n3.__h = [], t2.some(function(t3) {
            t3.call(n3);
        });
        } catch (t3) {
        c.__e(t3, n3.__v);
        }
    });
    }
    function R(t2, n2, e2, r2, o2, i2, u2, s2) {
    var a2, c2, f2, p2 = e2.props, d2 = n2.props, h2 = n2.type, _2 = 0;
    if (h2 === "svg" && (o2 = true), i2 != null) {
        for (; _2 < i2.length; _2++)
        if ((a2 = i2[_2]) && "setAttribute" in a2 == !!h2 && (h2 ? a2.localName === h2 : a2.nodeType === 3)) {
            t2 = a2, i2[_2] = null;
            break;
        }
    }
    if (t2 == null) {
        if (h2 === null)
        return document.createTextNode(d2);
        t2 = o2 ? document.createElementNS("http://www.w3.org/2000/svg", h2) : document.createElement(h2, d2.is && d2), i2 = null, s2 = false;
    }
    if (h2 === null)
        p2 === d2 || s2 && t2.data === d2 || (t2.data = d2);
    else {
        if (i2 = i2 && l.call(t2.childNodes), c2 = (p2 = e2.props || m).dangerouslySetInnerHTML, f2 = d2.dangerouslySetInnerHTML, !s2) {
        if (i2 != null)
            for (p2 = {}, _2 = 0; _2 < t2.attributes.length; _2++)
            p2[t2.attributes[_2].name] = t2.attributes[_2].value;
        (f2 || c2) && (f2 && (c2 && f2.__html == c2.__html || f2.__html === t2.innerHTML) || (t2.innerHTML = f2 && f2.__html || ""));
        }
        if (function(t3, n3, e3, r3, o3) {
        var i3;
        for (i3 in e3)
            i3 === "children" || i3 === "key" || i3 in n3 || j(t3, i3, null, e3[i3], r3);
        for (i3 in n3)
            o3 && typeof n3[i3] != "function" || i3 === "children" || i3 === "key" || i3 === "value" || i3 === "checked" || e3[i3] === n3[i3] || j(t3, i3, n3[i3], e3[i3], r3);
        }(t2, d2, p2, o2, s2), f2)
        n2.__k = [];
        else if (_2 = n2.props.children, T(t2, Array.isArray(_2) ? _2 : [_2], n2, e2, r2, o2 && h2 !== "foreignObject", i2, u2, i2 ? i2[0] : e2.__k && C(e2, 0), s2), i2 != null)
        for (_2 = i2.length; _2--; )
            i2[_2] != null && b(i2[_2]);
        s2 || ("value" in d2 && (_2 = d2.value) !== void 0 && (_2 !== t2.value || h2 === "progress" && !_2 || h2 === "option" && _2 !== p2.value) && j(t2, "value", _2, p2.value, false), "checked" in d2 && (_2 = d2.checked) !== void 0 && _2 !== t2.checked && j(t2, "checked", _2, p2.checked, false));
    }
    return t2;
    }
    function U(t2, n2, e2) {
    try {
        typeof t2 == "function" ? t2(n2) : t2.current = n2;
    } catch (t3) {
        c.__e(t3, e2);
    }
    }
    function W(t2, n2, e2) {
    var r2, o2;
    if (c.unmount && c.unmount(t2), (r2 = t2.ref) && (r2.current && r2.current !== t2.__e || U(r2, null, n2)), (r2 = t2.__c) != null) {
        if (r2.componentWillUnmount)
        try {
            r2.componentWillUnmount();
        } catch (t3) {
            c.__e(t3, n2);
        }
        r2.base = r2.__P = null, t2.__c = void 0;
    }
    if (r2 = t2.__k)
        for (o2 = 0; o2 < r2.length; o2++)
        r2[o2] && W(r2[o2], n2, e2 || typeof t2.type != "function");
    e2 || t2.__e == null || b(t2.__e), t2.__ = t2.__e = t2.__d = void 0;
    }
    function B(t2, n2, e2) {
    return this.constructor(t2, e2);
    }
    function q(t2, n2, e2) {
    var r2, o2, i2;
    c.__ && c.__(t2, n2), o2 = (r2 = typeof e2 == "function") ? null : e2 && e2.__k || n2.__k, i2 = [], F(n2, t2 = (!r2 && e2 || n2).__k = w(S, null, [t2]), o2 || m, m, n2.ownerSVGElement !== void 0, !r2 && e2 ? [e2] : o2 ? null : n2.firstChild ? l.call(n2.childNodes) : null, i2, !r2 && e2 ? e2 : o2 ? o2.__e : n2.firstChild, r2), O(i2, t2);
    }
    function z() {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function(t2) {
        var n2 = 16 * Math.random() | 0;
        return (t2 == "x" ? n2 : 3 & n2 | 8).toString(16);
    });
    }
    l = v.slice, c = { __e: function(t2, n2, e2, r2) {
    for (var o2, i2, u2; n2 = n2.__; )
        if ((o2 = n2.__c) && !o2.__)
        try {
            if ((i2 = o2.constructor) && i2.getDerivedStateFromError != null && (o2.setState(i2.getDerivedStateFromError(t2)), u2 = o2.__d), o2.componentDidCatch != null && (o2.componentDidCatch(t2, r2 || {}), u2 = o2.__d), u2)
            return o2.__E = o2;
        } catch (n3) {
            t2 = n3;
        }
    throw t2;
    } }, f = 0, p = function(t2) {
    return t2 != null && t2.constructor === void 0;
    }, N.prototype.setState = function(t2, n2) {
    var e2;
    e2 = this.__s != null && this.__s !== this.state ? this.__s : this.__s = y({}, this.state), typeof t2 == "function" && (t2 = t2(y({}, e2), this.props)), t2 && y(e2, t2), t2 != null && this.__v && (n2 && this._sb.push(n2), E(this));
    }, N.prototype.forceUpdate = function(t2) {
    this.__v && (this.__e = true, t2 && this.__h.push(t2), E(this));
    }, N.prototype.render = S, d = [], I.__r = 0, _ = 0;
    var V = /* @__PURE__ */ function() {
    function t2(t3) {
        this._id = void 0, this._id = t3 || z();
    }
    return n(t2, [{ key: "id", get: function() {
        return this._id;
    } }]), t2;
    }();
    function $(t2) {
    return w(t2.parentElement || "span", { dangerouslySetInnerHTML: { __html: t2.content } });
    }
    function G(t2, n2) {
    return w($, { content: t2, parentElement: n2 });
    }
    var K;
    var X = /* @__PURE__ */ function(t2) {
    function n2(n3) {
        var e3;
        return (e3 = t2.call(this) || this).data = void 0, e3.update(n3), e3;
    }
    r(n2, t2);
    var e2 = n2.prototype;
    return e2.cast = function(t3) {
        return t3 instanceof HTMLElement ? G(t3.outerHTML) : t3;
    }, e2.update = function(t3) {
        return this.data = this.cast(t3), this;
    }, n2;
    }(V);
    var Z = /* @__PURE__ */ function(t2) {
    function e2(n2) {
        var e3;
        return (e3 = t2.call(this) || this)._cells = void 0, e3.cells = n2 || [], e3;
    }
    r(e2, t2);
    var o2 = e2.prototype;
    return o2.cell = function(t3) {
        return this._cells[t3];
    }, o2.toArray = function() {
        return this.cells.map(function(t3) {
        return t3.data;
        });
    }, e2.fromCells = function(t3) {
        return new e2(t3.map(function(t4) {
        return new X(t4.data);
        }));
    }, n(e2, [{ key: "cells", get: function() {
        return this._cells;
    }, set: function(t3) {
        this._cells = t3;
    } }, { key: "length", get: function() {
        return this.cells.length;
    } }]), e2;
    }(V);
    var J = /* @__PURE__ */ function(t2) {
    function e2(n2) {
        var e3;
        return (e3 = t2.call(this) || this)._rows = void 0, e3._length = void 0, e3.rows = n2 instanceof Array ? n2 : n2 instanceof Z ? [n2] : [], e3;
    }
    return r(e2, t2), e2.prototype.toArray = function() {
        return this.rows.map(function(t3) {
        return t3.toArray();
        });
    }, e2.fromRows = function(t3) {
        return new e2(t3.map(function(t4) {
        return Z.fromCells(t4.cells);
        }));
    }, e2.fromArray = function(t3) {
        return new e2((t3 = function(t4) {
        return !t4[0] || t4[0] instanceof Array ? t4 : [t4];
        }(t3)).map(function(t4) {
        return new Z(t4.map(function(t5) {
            return new X(t5);
        }));
        }));
    }, n(e2, [{ key: "rows", get: function() {
        return this._rows;
    }, set: function(t3) {
        this._rows = t3;
    } }, { key: "length", get: function() {
        return this._length || this.rows.length;
    }, set: function(t3) {
        this._length = t3;
    } }]), e2;
    }(V);
    var Q = /* @__PURE__ */ function() {
    function t2() {
        this.callbacks = void 0;
    }
    var n2 = t2.prototype;
    return n2.init = function(t3) {
        this.callbacks || (this.callbacks = {}), t3 && !this.callbacks[t3] && (this.callbacks[t3] = []);
    }, n2.listeners = function() {
        return this.callbacks;
    }, n2.on = function(t3, n3) {
        return this.init(t3), this.callbacks[t3].push(n3), this;
    }, n2.off = function(t3, n3) {
        var e2 = t3;
        return this.init(), this.callbacks[e2] && this.callbacks[e2].length !== 0 ? (this.callbacks[e2] = this.callbacks[e2].filter(function(t4) {
        return t4 != n3;
        }), this) : this;
    }, n2.emit = function(t3) {
        var n3 = arguments, e2 = t3;
        return this.init(e2), this.callbacks[e2].length > 0 && (this.callbacks[e2].forEach(function(t4) {
        return t4.apply(void 0, [].slice.call(n3, 1));
        }), true);
    }, t2;
    }();
    !function(t2) {
    t2[t2.Initiator = 0] = "Initiator", t2[t2.ServerFilter = 1] = "ServerFilter", t2[t2.ServerSort = 2] = "ServerSort", t2[t2.ServerLimit = 3] = "ServerLimit", t2[t2.Extractor = 4] = "Extractor", t2[t2.Transformer = 5] = "Transformer", t2[t2.Filter = 6] = "Filter", t2[t2.Sort = 7] = "Sort", t2[t2.Limit = 8] = "Limit";
    }(K || (K = {}));
    var Y = /* @__PURE__ */ function(t2) {
    function e2(n2) {
        var e3;
        return (e3 = t2.call(this) || this).id = void 0, e3._props = void 0, e3._props = {}, e3.id = z(), n2 && e3.setProps(n2), e3;
    }
    r(e2, t2);
    var o2 = e2.prototype;
    return o2.process = function() {
        var t3 = [].slice.call(arguments);
        this.validateProps instanceof Function && this.validateProps.apply(this, t3), this.emit.apply(this, ["beforeProcess"].concat(t3));
        var n2 = this._process.apply(this, t3);
        return this.emit.apply(this, ["afterProcess"].concat(t3)), n2;
    }, o2.setProps = function(t3) {
        return Object.assign(this._props, t3), this.emit("propsUpdated", this), this;
    }, n(e2, [{ key: "props", get: function() {
        return this._props;
    } }]), e2;
    }(Q);
    var tt = /* @__PURE__ */ function(t2) {
    function e2() {
        return t2.apply(this, arguments) || this;
    }
    return r(e2, t2), e2.prototype._process = function(t3) {
        return this.props.keyword ? (n2 = String(this.props.keyword).trim(), e3 = this.props.columns, r2 = this.props.ignoreHiddenColumns, o2 = t3, i2 = this.props.selector, n2 = n2.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, "\\$&"), new J(o2.rows.filter(function(t4, o3) {
        return t4.cells.some(function(t5, u2) {
            if (!t5)
            return false;
            if (r2 && e3 && e3[u2] && typeof e3[u2] == "object" && e3[u2].hidden)
            return false;
            var s2 = "";
            if (typeof i2 == "function")
            s2 = i2(t5.data, o3, u2);
            else if (typeof t5.data == "object") {
            var a2 = t5.data;
            a2 && a2.props && a2.props.content && (s2 = a2.props.content);
            } else
            s2 = String(t5.data);
            return new RegExp(n2, "gi").test(s2);
        });
        }))) : t3;
        var n2, e3, r2, o2, i2;
    }, n(e2, [{ key: "type", get: function() {
        return K.Filter;
    } }]), e2;
    }(Y);
    function nt() {
    var t2 = "gridjs";
    return "" + t2 + [].slice.call(arguments).reduce(function(t3, n2) {
        return t3 + "-" + n2;
    }, "");
    }
    function et() {
    return [].slice.call(arguments).map(function(t2) {
        return t2 ? t2.toString() : "";
    }).filter(function(t2) {
        return t2;
    }).reduce(function(t2, n2) {
        return (t2 || "") + " " + n2;
    }, "").trim();
    }
    var rt;
    var ot;
    var it;
    var ut;
    var st = /* @__PURE__ */ function(t2) {
    function o2() {
        return t2.apply(this, arguments) || this;
    }
    return r(o2, t2), o2.prototype._process = function(t3) {
        if (!this.props.keyword)
        return t3;
        var n2 = {};
        return this.props.url && (n2.url = this.props.url(t3.url, this.props.keyword)), this.props.body && (n2.body = this.props.body(t3.body, this.props.keyword)), e({}, t3, n2);
    }, n(o2, [{ key: "type", get: function() {
        return K.ServerFilter;
    } }]), o2;
    }(Y);
    var at = 0;
    var lt = [];
    var ct = [];
    var ft = c.__b;
    var pt = c.__r;
    var dt = c.diffed;
    var ht = c.__c;
    var _t = c.unmount;
    function mt(t2, n2) {
    c.__h && c.__h(ot, t2, at || n2), at = 0;
    var e2 = ot.__H || (ot.__H = { __: [], __h: [] });
    return t2 >= e2.__.length && e2.__.push({ __V: ct }), e2.__[t2];
    }
    function vt(t2) {
    return at = 1, function(t3, n2, e2) {
        var r2 = mt(rt++, 2);
        if (r2.t = t3, !r2.__c && (r2.__ = [Pt(void 0, n2), function(t4) {
        var n3 = r2.__N ? r2.__N[0] : r2.__[0], e3 = r2.t(n3, t4);
        n3 !== e3 && (r2.__N = [e3, r2.__[1]], r2.__c.setState({}));
        }], r2.__c = ot, !ot.u)) {
        ot.u = true;
        var o2 = ot.shouldComponentUpdate;
        ot.shouldComponentUpdate = function(t4, n3, e3) {
            if (!r2.__c.__H)
            return true;
            var i2 = r2.__c.__H.__.filter(function(t5) {
            return t5.__c;
            });
            if (i2.every(function(t5) {
            return !t5.__N;
            }))
            return !o2 || o2.call(this, t4, n3, e3);
            var u2 = false;
            return i2.forEach(function(t5) {
            if (t5.__N) {
                var n4 = t5.__[0];
                t5.__ = t5.__N, t5.__N = void 0, n4 !== t5.__[0] && (u2 = true);
            }
            }), !(!u2 && r2.__c.props === t4) && (!o2 || o2.call(this, t4, n3, e3));
        };
        }
        return r2.__N || r2.__;
    }(Pt, t2);
    }
    function gt(t2, n2) {
    var e2 = mt(rt++, 3);
    !c.__s && Ct(e2.__H, n2) && (e2.__ = t2, e2.i = n2, ot.__H.__h.push(e2));
    }
    function yt(t2) {
    return at = 5, bt(function() {
        return { current: t2 };
    }, []);
    }
    function bt(t2, n2) {
    var e2 = mt(rt++, 7);
    return Ct(e2.__H, n2) ? (e2.__V = t2(), e2.i = n2, e2.__h = t2, e2.__V) : e2.__;
    }
    function wt() {
    for (var t2; t2 = lt.shift(); )
        if (t2.__P && t2.__H)
        try {
            t2.__H.__h.forEach(St), t2.__H.__h.forEach(Nt), t2.__H.__h = [];
        } catch (n2) {
            t2.__H.__h = [], c.__e(n2, t2.__v);
        }
    }
    c.__b = function(t2) {
    ot = null, ft && ft(t2);
    }, c.__r = function(t2) {
    pt && pt(t2), rt = 0;
    var n2 = (ot = t2.__c).__H;
    n2 && (it === ot ? (n2.__h = [], ot.__h = [], n2.__.forEach(function(t3) {
        t3.__N && (t3.__ = t3.__N), t3.__V = ct, t3.__N = t3.i = void 0;
    })) : (n2.__h.forEach(St), n2.__h.forEach(Nt), n2.__h = [])), it = ot;
    }, c.diffed = function(t2) {
    dt && dt(t2);
    var n2 = t2.__c;
    n2 && n2.__H && (n2.__H.__h.length && (lt.push(n2) !== 1 && ut === c.requestAnimationFrame || ((ut = c.requestAnimationFrame) || kt)(wt)), n2.__H.__.forEach(function(t3) {
        t3.i && (t3.__H = t3.i), t3.__V !== ct && (t3.__ = t3.__V), t3.i = void 0, t3.__V = ct;
    })), it = ot = null;
    }, c.__c = function(t2, n2) {
    n2.some(function(t3) {
        try {
        t3.__h.forEach(St), t3.__h = t3.__h.filter(function(t4) {
            return !t4.__ || Nt(t4);
        });
        } catch (e2) {
        n2.some(function(t4) {
            t4.__h && (t4.__h = []);
        }), n2 = [], c.__e(e2, t3.__v);
        }
    }), ht && ht(t2, n2);
    }, c.unmount = function(t2) {
    _t && _t(t2);
    var n2, e2 = t2.__c;
    e2 && e2.__H && (e2.__H.__.forEach(function(t3) {
        try {
        St(t3);
        } catch (t4) {
        n2 = t4;
        }
    }), e2.__H = void 0, n2 && c.__e(n2, e2.__v));
    };
    var xt = typeof requestAnimationFrame == "function";
    function kt(t2) {
    var n2, e2 = function() {
        clearTimeout(r2), xt && cancelAnimationFrame(n2), setTimeout(t2);
    }, r2 = setTimeout(e2, 100);
    xt && (n2 = requestAnimationFrame(e2));
    }
    function St(t2) {
    var n2 = ot, e2 = t2.__c;
    typeof e2 == "function" && (t2.__c = void 0, e2()), ot = n2;
    }
    function Nt(t2) {
    var n2 = ot;
    t2.__c = t2.__(), ot = n2;
    }
    function Ct(t2, n2) {
    return !t2 || t2.length !== n2.length || n2.some(function(n3, e2) {
        return n3 !== t2[e2];
    });
    }
    function Pt(t2, n2) {
    return typeof n2 == "function" ? n2(t2) : n2;
    }
    function Et() {
    return function(t2) {
        var n2 = ot.context[t2.__c], e2 = mt(rt++, 9);
        return e2.c = t2, n2 ? (e2.__ == null && (e2.__ = true, n2.sub(ot)), n2.props.value) : t2.__;
    }(cn);
    }
    var It = { search: { placeholder: "Type a keyword..." }, sort: { sortAsc: "Sort column ascending", sortDesc: "Sort column descending" }, pagination: { previous: "Previous", next: "Next", navigate: function(t2, n2) {
    return "Page " + t2 + " of " + n2;
    }, page: function(t2) {
    return "Page " + t2;
    }, showing: "Showing", of: "of", to: "to", results: "results" }, loading: "Loading...", noRecordsFound: "No matching records found", error: "An error happened while fetching the data" };
    var Tt = /* @__PURE__ */ function() {
    function t2(t3) {
        this._language = void 0, this._defaultLanguage = void 0, this._language = t3, this._defaultLanguage = It;
    }
    var n2 = t2.prototype;
    return n2.getString = function(t3, n3) {
        if (!n3 || !t3)
        return null;
        var e2 = t3.split("."), r2 = e2[0];
        if (n3[r2]) {
        var o2 = n3[r2];
        return typeof o2 == "string" ? function() {
            return o2;
        } : typeof o2 == "function" ? o2 : this.getString(e2.slice(1).join("."), o2);
        }
        return null;
    }, n2.translate = function(t3) {
        var n3, e2 = this.getString(t3, this._language);
        return (n3 = e2 || this.getString(t3, this._defaultLanguage)) ? n3.apply(void 0, [].slice.call(arguments, 1)) : t3;
    }, t2;
    }();
    function Lt() {
    var t2 = Et();
    return function(n2) {
        var e2;
        return (e2 = t2.translator).translate.apply(e2, [n2].concat([].slice.call(arguments, 1)));
    };
    }
    var At = function(t2) {
    return function(n2) {
        return e({}, n2, { search: { keyword: t2 } });
    };
    };
    function Ht() {
    return Et().store;
    }
    function jt(t2) {
    var n2 = Ht(), e2 = vt(t2(n2.getState())), r2 = e2[0], o2 = e2[1];
    return gt(function() {
        return n2.subscribe(function() {
        var e3 = t2(n2.getState());
        r2 !== e3 && o2(e3);
        });
    }, []), r2;
    }
    function Dt() {
    var t2, n2 = vt(void 0), e2 = n2[0], r2 = n2[1], o2 = Et(), i2 = o2.search, u2 = Lt(), s2 = Ht().dispatch, a2 = jt(function(t3) {
        return t3.search;
    });
    gt(function() {
        e2 && e2.setProps({ keyword: a2 == null ? void 0 : a2.keyword });
    }, [a2, e2]), gt(function() {
        r2(i2.server ? new st({ keyword: i2.keyword, url: i2.server.url, body: i2.server.body }) : new tt({ keyword: i2.keyword, columns: o2.header && o2.header.columns, ignoreHiddenColumns: i2.ignoreHiddenColumns || i2.ignoreHiddenColumns === void 0, selector: i2.selector })), i2.keyword && s2(At(i2.keyword));
    }, [i2]), gt(function() {
        return o2.pipeline.register(e2), function() {
        return o2.pipeline.unregister(e2);
        };
    }, [o2, e2]);
    var l2, c2, f2, p2 = function(t3, n3) {
        return at = 8, bt(function() {
        return t3;
        }, n3);
    }((l2 = function(t3) {
        t3.target instanceof HTMLInputElement && s2(At(t3.target.value));
    }, c2 = e2 instanceof st ? i2.debounceTimeout || 250 : 0, function() {
        var t3 = arguments;
        return new Promise(function(n3) {
        f2 && clearTimeout(f2), f2 = setTimeout(function() {
            return n3(l2.apply(void 0, [].slice.call(t3)));
        }, c2);
        });
    }), [i2, e2]);
    return w("div", { className: nt(et("search", (t2 = o2.className) == null ? void 0 : t2.search)) }, w("input", { type: "search", placeholder: u2("search.placeholder"), "aria-label": u2("search.placeholder"), onInput: p2, className: et(nt("input"), nt("search", "input")), value: (a2 == null ? void 0 : a2.keyword) || "" }));
    }
    var Mt = /* @__PURE__ */ function(t2) {
    function e2() {
        return t2.apply(this, arguments) || this;
    }
    r(e2, t2);
    var o2 = e2.prototype;
    return o2.validateProps = function() {
        if (isNaN(Number(this.props.limit)) || isNaN(Number(this.props.page)))
        throw Error("Invalid parameters passed");
    }, o2._process = function(t3) {
        var n2 = this.props.page;
        return new J(t3.rows.slice(n2 * this.props.limit, (n2 + 1) * this.props.limit));
    }, n(e2, [{ key: "type", get: function() {
        return K.Limit;
    } }]), e2;
    }(Y);
    var Ft = /* @__PURE__ */ function(t2) {
    function o2() {
        return t2.apply(this, arguments) || this;
    }
    return r(o2, t2), o2.prototype._process = function(t3) {
        var n2 = {};
        return this.props.url && (n2.url = this.props.url(t3.url, this.props.page, this.props.limit)), this.props.body && (n2.body = this.props.body(t3.body, this.props.page, this.props.limit)), e({}, t3, n2);
    }, n(o2, [{ key: "type", get: function() {
        return K.ServerLimit;
    } }]), o2;
    }(Y);
    function Ot() {
    var t2 = Et(), n2 = t2.pagination, e2 = n2.server, r2 = n2.summary, o2 = r2 === void 0 || r2, i2 = n2.nextButton, u2 = i2 === void 0 || i2, s2 = n2.prevButton, a2 = s2 === void 0 || s2, l2 = n2.buttonsCount, c2 = l2 === void 0 ? 3 : l2, f2 = n2.limit, p2 = f2 === void 0 ? 10 : f2, d2 = n2.page, h2 = d2 === void 0 ? 0 : d2, _2 = n2.resetPageOnUpdate, m2 = _2 === void 0 || _2, v2 = yt(null), g2 = vt(h2), y2 = g2[0], b2 = g2[1], x2 = vt(0), k2 = x2[0], N2 = x2[1], C2 = Lt();
    gt(function() {
        return v2.current = e2 ? new Ft({ limit: p2, page: y2, url: e2.url, body: e2.body }) : new Mt({ limit: p2, page: y2 }), v2.current instanceof Ft ? t2.pipeline.on("afterProcess", function(t3) {
        return N2(t3.length);
        }) : v2.current instanceof Mt && v2.current.on("beforeProcess", function(t3) {
        return N2(t3.length);
        }), t2.pipeline.on("updated", P2), t2.pipeline.register(v2.current), t2.pipeline.on("error", function() {
        N2(0), b2(0);
        }), function() {
        t2.pipeline.unregister(v2.current), t2.pipeline.off("updated", P2);
        };
    }, []);
    var P2 = function(t3) {
        m2 && t3 !== v2.current && b2(0);
    }, E2 = function() {
        return Math.ceil(k2 / p2);
    }, I2 = function(t3) {
        if (t3 >= E2() || t3 < 0 || t3 === y2)
        return null;
        b2(t3), v2.current.setProps({ page: t3 });
    };
    return w("div", { className: et(nt("pagination"), t2.className.pagination) }, w(S, null, o2 && k2 > 0 && w("div", { role: "status", "aria-live": "polite", className: et(nt("summary"), t2.className.paginationSummary), title: C2("pagination.navigate", y2 + 1, E2()) }, C2("pagination.showing"), " ", w("b", null, C2("" + (y2 * p2 + 1))), " ", C2("pagination.to"), " ", w("b", null, C2("" + Math.min((y2 + 1) * p2, k2))), " ", C2("pagination.of"), " ", w("b", null, C2("" + k2)), " ", C2("pagination.results"))), w("div", { className: nt("pages") }, a2 && w("button", { tabIndex: 0, role: "button", disabled: y2 === 0, onClick: function() {
        return I2(y2 - 1);
    }, title: C2("pagination.previous"), "aria-label": C2("pagination.previous"), className: et(t2.className.paginationButton, t2.className.paginationButtonPrev) }, C2("pagination.previous")), function() {
        if (c2 <= 0)
        return null;
        var n3 = Math.min(E2(), c2), e3 = Math.min(y2, Math.floor(n3 / 2));
        return y2 + Math.floor(n3 / 2) >= E2() && (e3 = n3 - (E2() - y2)), w(S, null, E2() > n3 && y2 - e3 > 0 && w(S, null, w("button", { tabIndex: 0, role: "button", onClick: function() {
        return I2(0);
        }, title: C2("pagination.firstPage"), "aria-label": C2("pagination.firstPage"), className: t2.className.paginationButton }, C2("1")), w("button", { tabIndex: -1, className: et(nt("spread"), t2.className.paginationButton) }, "...")), Array.from(Array(n3).keys()).map(function(t3) {
        return y2 + (t3 - e3);
        }).map(function(n4) {
        return w("button", { tabIndex: 0, role: "button", onClick: function() {
            return I2(n4);
        }, className: et(y2 === n4 ? et(nt("currentPage"), t2.className.paginationButtonCurrent) : null, t2.className.paginationButton), title: C2("pagination.page", n4 + 1), "aria-label": C2("pagination.page", n4 + 1) }, C2("" + (n4 + 1)));
        }), E2() > n3 && E2() > y2 + e3 + 1 && w(S, null, w("button", { tabIndex: -1, className: et(nt("spread"), t2.className.paginationButton) }, "..."), w("button", { tabIndex: 0, role: "button", onClick: function() {
        return I2(E2() - 1);
        }, title: C2("pagination.page", E2()), "aria-label": C2("pagination.page", E2()), className: t2.className.paginationButton }, C2("" + E2()))));
    }(), u2 && w("button", { tabIndex: 0, role: "button", disabled: E2() === y2 + 1 || E2() === 0, onClick: function() {
        return I2(y2 + 1);
    }, title: C2("pagination.next"), "aria-label": C2("pagination.next"), className: et(t2.className.paginationButton, t2.className.paginationButtonNext) }, C2("pagination.next"))));
    }
    function Rt(t2, n2) {
    return typeof t2 == "string" ? t2.indexOf("%") > -1 ? n2 / 100 * parseInt(t2, 10) : parseInt(t2, 10) : t2;
    }
    function Ut(t2) {
    return t2 ? Math.floor(t2) + "px" : "";
    }
    function Wt(t2) {
    var n2 = t2.tableRef.cloneNode(true);
    return n2.style.position = "absolute", n2.style.width = "100%", n2.style.zIndex = "-2147483640", n2.style.visibility = "hidden", w("div", { ref: function(t3) {
        t3 && t3.appendChild(n2);
    } });
    }
    function Bt(t2) {
    if (!t2)
        return "";
    var n2 = t2.split(" ");
    return n2.length === 1 && /([a-z][A-Z])+/g.test(t2) ? t2 : n2.map(function(t3, n3) {
        return n3 == 0 ? t3.toLowerCase() : t3.charAt(0).toUpperCase() + t3.slice(1).toLowerCase();
    }).join("");
    }
    var qt;
    var zt = new (/* @__PURE__ */ function() {
    function t2() {
    }
    var n2 = t2.prototype;
    return n2.format = function(t3, n3) {
        return "[Grid.js] [" + n3.toUpperCase() + "]: " + t3;
    }, n2.error = function(t3, n3) {
        n3 === void 0 && (n3 = false);
        var e2 = this.format(t3, "error");
        if (n3)
        throw Error(e2);
        console.error(e2);
    }, n2.warn = function(t3) {
        console.warn(this.format(t3, "warn"));
    }, n2.info = function(t3) {
        console.info(this.format(t3, "info"));
    }, t2;
    }())();
    !function(t2) {
    t2[t2.Header = 0] = "Header", t2[t2.Footer = 1] = "Footer", t2[t2.Cell = 2] = "Cell";
    }(qt || (qt = {}));
    var Vt = /* @__PURE__ */ function() {
    function t2() {
        this.plugins = void 0, this.plugins = [];
    }
    var n2 = t2.prototype;
    return n2.get = function(t3) {
        return this.plugins.find(function(n3) {
        return n3.id === t3;
        });
    }, n2.add = function(t3) {
        return t3.id ? this.get(t3.id) ? (zt.error("Duplicate plugin ID: " + t3.id), this) : (this.plugins.push(t3), this) : (zt.error("Plugin ID cannot be empty"), this);
    }, n2.remove = function(t3) {
        var n3 = this.get(t3);
        return n3 && this.plugins.splice(this.plugins.indexOf(n3), 1), this;
    }, n2.list = function(t3) {
        var n3;
        return n3 = t3 != null || t3 != null ? this.plugins.filter(function(n4) {
        return n4.position === t3;
        }) : this.plugins, n3.sort(function(t4, n4) {
        return t4.order && n4.order ? t4.order - n4.order : 1;
        });
    }, t2;
    }();
    function $t(t2) {
    var n2 = this, r2 = Et();
    if (t2.pluginId) {
        var o2 = r2.plugin.get(t2.pluginId);
        return o2 ? w(S, {}, w(o2.component, e({ plugin: o2 }, t2.props))) : null;
    }
    return t2.position !== void 0 ? w(S, {}, r2.plugin.list(t2.position).map(function(t3) {
        return w(t3.component, e({ plugin: t3 }, n2.props.props));
    })) : null;
    }
    var Gt = /* @__PURE__ */ function(t2) {
    function o2() {
        var n2;
        return (n2 = t2.call(this) || this)._columns = void 0, n2._columns = [], n2;
    }
    r(o2, t2);
    var i2 = o2.prototype;
    return i2.adjustWidth = function(t3, n2, r2) {
        var i3 = t3.container, u2 = t3.autoWidth;
        if (!i3)
        return this;
        var a2 = i3.clientWidth, l2 = {};
        n2.current && u2 && (q(w(Wt, { tableRef: n2.current }), r2.current), l2 = function(t4) {
        var n3 = t4.querySelector("table");
        if (!n3)
            return {};
        var r3 = n3.className, o3 = n3.style.cssText;
        n3.className = r3 + " " + nt("shadowTable"), n3.style.tableLayout = "auto", n3.style.width = "auto", n3.style.padding = "0", n3.style.margin = "0", n3.style.border = "none", n3.style.outline = "none";
        var i4 = Array.from(n3.parentNode.querySelectorAll("thead th")).reduce(function(t5, n4) {
            var r4;
            return n4.style.width = n4.clientWidth + "px", e(((r4 = {})[n4.getAttribute("data-column-id")] = { minWidth: n4.clientWidth }, r4), t5);
        }, {});
        return n3.className = r3, n3.style.cssText = o3, n3.style.tableLayout = "auto", Array.from(n3.parentNode.querySelectorAll("thead th")).reduce(function(t5, n4) {
            return t5[n4.getAttribute("data-column-id")].width = n4.clientWidth, t5;
        }, i4);
        }(r2.current));
        for (var c2, f2 = s(o2.tabularFormat(this.columns).reduce(function(t4, n3) {
        return t4.concat(n3);
        }, [])); !(c2 = f2()).done; ) {
        var p2 = c2.value;
        p2.columns && p2.columns.length > 0 || (!p2.width && u2 ? p2.id in l2 && (p2.width = Ut(l2[p2.id].width), p2.minWidth = Ut(l2[p2.id].minWidth)) : p2.width = Ut(Rt(p2.width, a2)));
        }
        return n2.current && u2 && q(null, r2.current), this;
    }, i2.setSort = function(t3, n2) {
        for (var r2, o3 = s(n2 || this.columns || []); !(r2 = o3()).done; ) {
        var i3 = r2.value;
        i3.columns && i3.columns.length > 0 ? i3.sort = void 0 : i3.sort === void 0 && t3 ? i3.sort = {} : i3.sort ? typeof i3.sort == "object" && (i3.sort = e({}, i3.sort)) : i3.sort = void 0, i3.columns && this.setSort(t3, i3.columns);
        }
    }, i2.setResizable = function(t3, n2) {
        for (var e2, r2 = s(n2 || this.columns || []); !(e2 = r2()).done; ) {
        var o3 = e2.value;
        o3.resizable === void 0 && (o3.resizable = t3), o3.columns && this.setResizable(t3, o3.columns);
        }
    }, i2.setID = function(t3) {
        for (var n2, e2 = s(t3 || this.columns || []); !(n2 = e2()).done; ) {
        var r2 = n2.value;
        r2.id || typeof r2.name != "string" || (r2.id = Bt(r2.name)), r2.id || zt.error('Could not find a valid ID for one of the columns. Make sure a valid "id" is set for all columns.'), r2.columns && this.setID(r2.columns);
        }
    }, i2.populatePlugins = function(t3, n2) {
        for (var r2, o3 = s(n2); !(r2 = o3()).done; ) {
        var i3 = r2.value;
        i3.plugin !== void 0 && t3.add(e({ id: i3.id }, i3.plugin, { position: qt.Cell }));
        }
    }, o2.fromColumns = function(t3) {
        for (var n2, e2 = new o2(), r2 = s(t3); !(n2 = r2()).done; ) {
        var i3 = n2.value;
        if (typeof i3 == "string" || p(i3))
            e2.columns.push({ name: i3 });
        else if (typeof i3 == "object") {
            var u2 = i3;
            u2.columns && (u2.columns = o2.fromColumns(u2.columns).columns), typeof u2.plugin == "object" && u2.data === void 0 && (u2.data = null), e2.columns.push(i3);
        }
        }
        return e2;
    }, o2.createFromConfig = function(t3) {
        var n2 = new o2();
        return t3.from ? n2.columns = o2.fromHTMLTable(t3.from).columns : t3.columns ? n2.columns = o2.fromColumns(t3.columns).columns : !t3.data || typeof t3.data[0] != "object" || t3.data[0] instanceof Array || (n2.columns = Object.keys(t3.data[0]).map(function(t4) {
        return { name: t4 };
        })), n2.columns.length ? (n2.setID(), n2.setSort(t3.sort), n2.setResizable(t3.resizable), n2.populatePlugins(t3.plugin, n2.columns), n2) : null;
    }, o2.fromHTMLTable = function(t3) {
        for (var n2, e2 = new o2(), r2 = s(t3.querySelector("thead").querySelectorAll("th")); !(n2 = r2()).done; ) {
        var i3 = n2.value;
        e2.columns.push({ name: i3.innerHTML, width: i3.width });
        }
        return e2;
    }, o2.tabularFormat = function(t3) {
        var n2 = [], e2 = t3 || [], r2 = [];
        if (e2 && e2.length) {
        n2.push(e2);
        for (var o3, i3 = s(e2); !(o3 = i3()).done; ) {
            var u2 = o3.value;
            u2.columns && u2.columns.length && (r2 = r2.concat(u2.columns));
        }
        r2.length && (n2 = n2.concat(this.tabularFormat(r2)));
        }
        return n2;
    }, o2.leafColumns = function(t3) {
        var n2 = [], e2 = t3 || [];
        if (e2 && e2.length)
        for (var r2, o3 = s(e2); !(r2 = o3()).done; ) {
            var i3 = r2.value;
            i3.columns && i3.columns.length !== 0 || n2.push(i3), i3.columns && (n2 = n2.concat(this.leafColumns(i3.columns)));
        }
        return n2;
    }, o2.maximumDepth = function(t3) {
        return this.tabularFormat([t3]).length - 1;
    }, n(o2, [{ key: "columns", get: function() {
        return this._columns;
    }, set: function(t3) {
        this._columns = t3;
    } }, { key: "visibleColumns", get: function() {
        return this._columns.filter(function(t3) {
        return !t3.hidden;
        });
    } }]), o2;
    }(V);
    var Kt = function() {
    };
    var Xt = /* @__PURE__ */ function(t2) {
    function n2(n3) {
        var e3;
        return (e3 = t2.call(this) || this).data = void 0, e3.set(n3), e3;
    }
    r(n2, t2);
    var e2 = n2.prototype;
    return e2.get = function() {
        try {
        return Promise.resolve(this.data()).then(function(t3) {
            return { data: t3, total: t3.length };
        });
        } catch (t3) {
        return Promise.reject(t3);
        }
    }, e2.set = function(t3) {
        return t3 instanceof Array ? this.data = function() {
        return t3;
        } : t3 instanceof Function && (this.data = t3), this;
    }, n2;
    }(Kt);
    var Zt = /* @__PURE__ */ function(t2) {
    function n2(n3) {
        var e2;
        return (e2 = t2.call(this) || this).options = void 0, e2.options = n3, e2;
    }
    r(n2, t2);
    var o2 = n2.prototype;
    return o2.handler = function(t3) {
        return typeof this.options.handle == "function" ? this.options.handle(t3) : t3.ok ? t3.json() : (zt.error("Could not fetch data: " + t3.status + " - " + t3.statusText, true), null);
    }, o2.get = function(t3) {
        var n3 = e({}, this.options, t3);
        return typeof n3.data == "function" ? n3.data(n3) : fetch(n3.url, n3).then(this.handler.bind(this)).then(function(t4) {
        return { data: n3.then(t4), total: typeof n3.total == "function" ? n3.total(t4) : void 0 };
        });
    }, n2;
    }(Kt);
    var Jt = /* @__PURE__ */ function() {
    function t2() {
    }
    return t2.createFromConfig = function(t3) {
        var n2 = null;
        return t3.data && (n2 = new Xt(t3.data)), t3.from && (n2 = new Xt(this.tableElementToArray(t3.from)), t3.from.style.display = "none"), t3.server && (n2 = new Zt(t3.server)), n2 || zt.error("Could not determine the storage type", true), n2;
    }, t2.tableElementToArray = function(t3) {
        for (var n2, e2, r2 = [], o2 = s(t3.querySelector("tbody").querySelectorAll("tr")); !(n2 = o2()).done; ) {
        for (var i2, u2 = [], a2 = s(n2.value.querySelectorAll("td")); !(i2 = a2()).done; ) {
            var l2 = i2.value;
            l2.childNodes.length === 1 && l2.childNodes[0].nodeType === Node.TEXT_NODE ? u2.push((e2 = l2.innerHTML, new DOMParser().parseFromString(e2, "text/html").documentElement.textContent)) : u2.push(G(l2.innerHTML));
        }
        r2.push(u2);
        }
        return r2;
    }, t2;
    }();
    var Qt = typeof Symbol != "undefined" ? Symbol.iterator || (Symbol.iterator = Symbol("Symbol.iterator")) : "@@iterator";
    function Yt(t2, n2, e2) {
    if (!t2.s) {
        if (e2 instanceof tn) {
        if (!e2.s)
            return void (e2.o = Yt.bind(null, t2, n2));
        1 & n2 && (n2 = e2.s), e2 = e2.v;
        }
        if (e2 && e2.then)
        return void e2.then(Yt.bind(null, t2, n2), Yt.bind(null, t2, 2));
        t2.s = n2, t2.v = e2;
        var r2 = t2.o;
        r2 && r2(t2);
    }
    }
    var tn = /* @__PURE__ */ function() {
    function t2() {
    }
    return t2.prototype.then = function(n2, e2) {
        var r2 = new t2(), o2 = this.s;
        if (o2) {
        var i2 = 1 & o2 ? n2 : e2;
        if (i2) {
            try {
            Yt(r2, 1, i2(this.v));
            } catch (t3) {
            Yt(r2, 2, t3);
            }
            return r2;
        }
        return this;
        }
        return this.o = function(t3) {
        try {
            var o3 = t3.v;
            1 & t3.s ? Yt(r2, 1, n2 ? n2(o3) : o3) : e2 ? Yt(r2, 1, e2(o3)) : Yt(r2, 2, o3);
        } catch (t4) {
            Yt(r2, 2, t4);
        }
        }, r2;
    }, t2;
    }();
    function nn(t2) {
    return t2 instanceof tn && 1 & t2.s;
    }
    var en = /* @__PURE__ */ function(t2) {
    function e2(n2) {
        var e3;
        return (e3 = t2.call(this) || this)._steps = /* @__PURE__ */ new Map(), e3.cache = /* @__PURE__ */ new Map(), e3.lastProcessorIndexUpdated = -1, n2 && n2.forEach(function(t3) {
        return e3.register(t3);
        }), e3;
    }
    r(e2, t2);
    var o2 = e2.prototype;
    return o2.clearCache = function() {
        this.cache = /* @__PURE__ */ new Map(), this.lastProcessorIndexUpdated = -1;
    }, o2.register = function(t3, n2) {
        if (n2 === void 0 && (n2 = null), t3) {
        if (t3.type === null)
            throw Error("Processor type is not defined");
        t3.on("propsUpdated", this.processorPropsUpdated.bind(this)), this.addProcessorByPriority(t3, n2), this.afterRegistered(t3);
        }
    }, o2.unregister = function(t3) {
        if (t3) {
        var n2 = this._steps.get(t3.type);
        n2 && n2.length && (this._steps.set(t3.type, n2.filter(function(n3) {
            return n3 != t3;
        })), this.emit("updated", t3));
        }
    }, o2.addProcessorByPriority = function(t3, n2) {
        var e3 = this._steps.get(t3.type);
        if (!e3) {
        var r2 = [];
        this._steps.set(t3.type, r2), e3 = r2;
        }
        if (n2 === null || n2 < 0)
        e3.push(t3);
        else if (e3[n2]) {
        var o3 = e3.slice(0, n2 - 1), i2 = e3.slice(n2 + 1);
        this._steps.set(t3.type, o3.concat(t3).concat(i2));
        } else
        e3[n2] = t3;
    }, o2.getStepsByType = function(t3) {
        return this.steps.filter(function(n2) {
        return n2.type === t3;
        });
    }, o2.getSortedProcessorTypes = function() {
        return Object.keys(K).filter(function(t3) {
        return !isNaN(Number(t3));
        }).map(function(t3) {
        return Number(t3);
        });
    }, o2.process = function(t3) {
        try {
        var n2 = function(t4) {
            return e3.lastProcessorIndexUpdated = o3.length, e3.emit("afterProcess", i2), i2;
        }, e3 = this, r2 = e3.lastProcessorIndexUpdated, o3 = e3.steps, i2 = t3, u2 = function(t4, n3) {
            try {
            var u3 = function(t5, n4, e4) {
                if (typeof t5[Qt] == "function") {
                var r3, o4, i3, u4 = t5[Qt]();
                if (function t6(e5) {
                    try {
                    for (; !(r3 = u4.next()).done; )
                        if ((e5 = n4(r3.value)) && e5.then) {
                        if (!nn(e5))
                            return void e5.then(t6, i3 || (i3 = Yt.bind(null, o4 = new tn(), 2)));
                        e5 = e5.v;
                        }
                    o4 ? Yt(o4, 1, e5) : o4 = e5;
                    } catch (t7) {
                    Yt(o4 || (o4 = new tn()), 2, t7);
                    }
                }(), u4.return) {
                    var s2 = function(t6) {
                    try {
                        r3.done || u4.return();
                    } catch (t7) {
                    }
                    return t6;
                    };
                    if (o4 && o4.then)
                    return o4.then(s2, function(t6) {
                        throw s2(t6);
                    });
                    s2();
                }
                return o4;
                }
                if (!("length" in t5))
                throw new TypeError("Object is not iterable");
                for (var a2 = [], l2 = 0; l2 < t5.length; l2++)
                a2.push(t5[l2]);
                return function(t6, n5, e5) {
                var r4, o5, i4 = -1;
                return function e6(u5) {
                    try {
                    for (; ++i4 < t6.length; )
                        if ((u5 = n5(i4)) && u5.then) {
                        if (!nn(u5))
                            return void u5.then(e6, o5 || (o5 = Yt.bind(null, r4 = new tn(), 2)));
                        u5 = u5.v;
                        }
                    r4 ? Yt(r4, 1, u5) : r4 = u5;
                    } catch (t7) {
                    Yt(r4 || (r4 = new tn()), 2, t7);
                    }
                }(), r4;
                }(a2, function(t6) {
                return n4(a2[t6]);
                });
            }(o3, function(t5) {
                var n4 = e3.findProcessorIndexByID(t5.id), o4 = function() {
                if (n4 >= r2)
                    return Promise.resolve(t5.process(i2)).then(function(n5) {
                    e3.cache.set(t5.id, i2 = n5);
                    });
                i2 = e3.cache.get(t5.id);
                }();
                if (o4 && o4.then)
                return o4.then(function() {
                });
            });
            } catch (t5) {
            return n3(t5);
            }
            return u3 && u3.then ? u3.then(void 0, n3) : u3;
        }(0, function(t4) {
            throw zt.error(t4), e3.emit("error", i2), t4;
        });
        return Promise.resolve(u2 && u2.then ? u2.then(n2) : n2());
        } catch (t4) {
        return Promise.reject(t4);
        }
    }, o2.findProcessorIndexByID = function(t3) {
        return this.steps.findIndex(function(n2) {
        return n2.id == t3;
        });
    }, o2.setLastProcessorIndex = function(t3) {
        var n2 = this.findProcessorIndexByID(t3.id);
        this.lastProcessorIndexUpdated > n2 && (this.lastProcessorIndexUpdated = n2);
    }, o2.processorPropsUpdated = function(t3) {
        this.setLastProcessorIndex(t3), this.emit("propsUpdated"), this.emit("updated", t3);
    }, o2.afterRegistered = function(t3) {
        this.setLastProcessorIndex(t3), this.emit("afterRegister"), this.emit("updated", t3);
    }, n(e2, [{ key: "steps", get: function() {
        for (var t3, n2 = [], e3 = s(this.getSortedProcessorTypes()); !(t3 = e3()).done; ) {
        var r2 = this._steps.get(t3.value);
        r2 && r2.length && (n2 = n2.concat(r2));
        }
        return n2.filter(function(t4) {
        return t4;
        });
    } }]), e2;
    }(Q);
    var rn = /* @__PURE__ */ function(t2) {
    function e2() {
        return t2.apply(this, arguments) || this;
    }
    return r(e2, t2), e2.prototype._process = function(t3) {
        try {
        return Promise.resolve(this.props.storage.get(t3));
        } catch (t4) {
        return Promise.reject(t4);
        }
    }, n(e2, [{ key: "type", get: function() {
        return K.Extractor;
    } }]), e2;
    }(Y);
    var on = /* @__PURE__ */ function(t2) {
    function e2() {
        return t2.apply(this, arguments) || this;
    }
    return r(e2, t2), e2.prototype._process = function(t3) {
        var n2 = J.fromArray(t3.data);
        return n2.length = t3.total, n2;
    }, n(e2, [{ key: "type", get: function() {
        return K.Transformer;
    } }]), e2;
    }(Y);
    var un = /* @__PURE__ */ function(t2) {
    function o2() {
        return t2.apply(this, arguments) || this;
    }
    return r(o2, t2), o2.prototype._process = function() {
        return Object.entries(this.props.serverStorageOptions).filter(function(t3) {
        return typeof t3[1] != "function";
        }).reduce(function(t3, n2) {
        var r2;
        return e({}, t3, ((r2 = {})[n2[0]] = n2[1], r2));
        }, {});
    }, n(o2, [{ key: "type", get: function() {
        return K.Initiator;
    } }]), o2;
    }(Y);
    var sn = /* @__PURE__ */ function(t2) {
    function e2() {
        return t2.apply(this, arguments) || this;
    }
    r(e2, t2);
    var o2 = e2.prototype;
    return o2.castData = function(t3) {
        if (!t3 || !t3.length)
        return [];
        if (!this.props.header || !this.props.header.columns)
        return t3;
        var n2 = Gt.leafColumns(this.props.header.columns);
        return t3[0] instanceof Array ? t3.map(function(t4) {
        var e3 = 0;
        return n2.map(function(n3, r2) {
            return n3.data !== void 0 ? (e3++, typeof n3.data == "function" ? n3.data(t4) : n3.data) : t4[r2 - e3];
        });
        }) : typeof t3[0] != "object" || t3[0] instanceof Array ? [] : t3.map(function(t4) {
        return n2.map(function(n3, e3) {
            return n3.data !== void 0 ? typeof n3.data == "function" ? n3.data(t4) : n3.data : n3.id ? t4[n3.id] : (zt.error("Could not find the correct cell for column at position " + e3 + ".\n                          Make sure either 'id' or 'selector' is defined for all columns."), null);
        });
        });
    }, o2._process = function(t3) {
        return { data: this.castData(t3.data), total: t3.total };
    }, n(e2, [{ key: "type", get: function() {
        return K.Transformer;
    } }]), e2;
    }(Y);
    var an = /* @__PURE__ */ function() {
    function t2() {
    }
    return t2.createFromConfig = function(t3) {
        var n2 = new en();
        return t3.storage instanceof Zt && n2.register(new un({ serverStorageOptions: t3.server })), n2.register(new rn({ storage: t3.storage })), n2.register(new sn({ header: t3.header })), n2.register(new on()), n2;
    }, t2;
    }();
    var ln = function(t2) {
    var n2 = this;
    this.state = void 0, this.listeners = [], this.isDispatching = false, this.getState = function() {
        return n2.state;
    }, this.getListeners = function() {
        return n2.listeners;
    }, this.dispatch = function(t3) {
        if (typeof t3 != "function")
        throw new Error("Reducer is not a function");
        if (n2.isDispatching)
        throw new Error("Reducers may not dispatch actions");
        n2.isDispatching = true;
        var e2 = n2.state;
        try {
        n2.state = t3(n2.state);
        } finally {
        n2.isDispatching = false;
        }
        for (var r2, o2 = s(n2.listeners); !(r2 = o2()).done; )
        (0, r2.value)(n2.state, e2);
        return n2.state;
    }, this.subscribe = function(t3) {
        if (typeof t3 != "function")
        throw new Error("Listener is not a function");
        return n2.listeners = [].concat(n2.listeners, [t3]), function() {
        return n2.listeners = n2.listeners.filter(function(n3) {
            return n3 !== t3;
        });
        };
    }, this.state = t2;
    };
    var cn = function(t2, n2) {
    var e2 = { __c: n2 = "__cC" + _++, __: null, Consumer: function(t3, n3) {
        return t3.children(n3);
    }, Provider: function(t3) {
        var e3, r2;
        return this.getChildContext || (e3 = [], (r2 = {})[n2] = this, this.getChildContext = function() {
        return r2;
        }, this.shouldComponentUpdate = function(t4) {
        this.props.value !== t4.value && e3.some(E);
        }, this.sub = function(t4) {
        e3.push(t4);
        var n3 = t4.componentWillUnmount;
        t4.componentWillUnmount = function() {
            e3.splice(e3.indexOf(t4), 1), n3 && n3.call(t4);
        };
        }), t3.children;
    } };
    return e2.Provider.__ = e2.Consumer.contextType = e2;
    }();
    var fn = /* @__PURE__ */ function() {
    function t2() {
        Object.assign(this, t2.defaultConfig());
    }
    var n2 = t2.prototype;
    return n2.assign = function(t3) {
        return Object.assign(this, t3);
    }, n2.update = function(n3) {
        return n3 ? (this.assign(t2.fromPartialConfig(e({}, this, n3))), this) : this;
    }, t2.defaultConfig = function() {
        return { store: new ln({ status: a.Init, header: void 0, data: null }), plugin: new Vt(), tableRef: { current: null }, width: "100%", height: "auto", autoWidth: true, style: {}, className: {} };
    }, t2.fromPartialConfig = function(n3) {
        var e2 = new t2().assign(n3);
        return typeof n3.sort == "boolean" && n3.sort && e2.assign({ sort: { multiColumn: true } }), e2.assign({ header: Gt.createFromConfig(e2) }), e2.assign({ storage: Jt.createFromConfig(e2) }), e2.assign({ pipeline: an.createFromConfig(e2) }), e2.assign({ translator: new Tt(e2.language) }), e2.search && e2.plugin.add({ id: "search", position: qt.Header, component: Dt }), e2.pagination && e2.plugin.add({ id: "pagination", position: qt.Footer, component: Ot }), e2.plugins && e2.plugins.forEach(function(t3) {
        return e2.plugin.add(t3);
        }), e2;
    }, t2;
    }();
    function pn(t2) {
    var n2, r2 = Et();
    return w("td", e({ role: t2.role, colSpan: t2.colSpan, "data-column-id": t2.column && t2.column.id, className: et(nt("td"), t2.className, r2.className.td), style: e({}, t2.style, r2.style.td), onClick: function(n3) {
        t2.messageCell || r2.eventEmitter.emit("cellClick", n3, t2.cell, t2.column, t2.row);
    } }, (n2 = t2.column) ? typeof n2.attributes == "function" ? n2.attributes(t2.cell.data, t2.row, t2.column) : n2.attributes : {}), t2.column && typeof t2.column.formatter == "function" ? t2.column.formatter(t2.cell.data, t2.row, t2.column) : t2.column && t2.column.plugin ? w($t, { pluginId: t2.column.id, props: { column: t2.column, cell: t2.cell, row: t2.row } }) : t2.cell.data);
    }
    function dn(t2) {
    var n2 = Et(), e2 = jt(function(t3) {
        return t3.header;
    });
    return w("tr", { className: et(nt("tr"), n2.className.tr), onClick: function(e3) {
        t2.messageRow || n2.eventEmitter.emit("rowClick", e3, t2.row);
    } }, t2.children ? t2.children : t2.row.cells.map(function(n3, r2) {
        var o2 = function(t3) {
        if (e2) {
            var n4 = Gt.leafColumns(e2.columns);
            if (n4)
            return n4[t3];
        }
        return null;
        }(r2);
        return o2 && o2.hidden ? null : w(pn, { key: n3.id, cell: n3, row: t2.row, column: o2 });
    }));
    }
    function hn(t2) {
    return w(dn, { messageRow: true }, w(pn, { role: "alert", colSpan: t2.colSpan, messageCell: true, cell: new X(t2.message), className: et(nt("message"), t2.className ? t2.className : null) }));
    }
    function _n() {
    var t2 = Et(), n2 = jt(function(t3) {
        return t3.data;
    }), e2 = jt(function(t3) {
        return t3.status;
    }), r2 = jt(function(t3) {
        return t3.header;
    }), o2 = Lt(), i2 = function() {
        return r2 ? r2.visibleColumns.length : 0;
    };
    return w("tbody", { className: et(nt("tbody"), t2.className.tbody) }, n2 && n2.rows.map(function(t3) {
        return w(dn, { key: t3.id, row: t3 });
    }), e2 === a.Loading && (!n2 || n2.length === 0) && w(hn, { message: o2("loading"), colSpan: i2(), className: et(nt("loading"), t2.className.loading) }), e2 === a.Rendered && n2 && n2.length === 0 && w(hn, { message: o2("noRecordsFound"), colSpan: i2(), className: et(nt("notfound"), t2.className.notfound) }), e2 === a.Error && w(hn, { message: o2("error"), colSpan: i2(), className: et(nt("error"), t2.className.error) }));
    }
    var mn = /* @__PURE__ */ function(t2) {
    function e2() {
        return t2.apply(this, arguments) || this;
    }
    r(e2, t2);
    var o2 = e2.prototype;
    return o2.validateProps = function() {
        for (var t3, n2 = s(this.props.columns); !(t3 = n2()).done; ) {
        var e3 = t3.value;
        e3.direction === void 0 && (e3.direction = 1), e3.direction !== 1 && e3.direction !== -1 && zt.error("Invalid sort direction " + e3.direction);
        }
    }, o2.compare = function(t3, n2) {
        return t3 > n2 ? 1 : t3 < n2 ? -1 : 0;
    }, o2.compareWrapper = function(t3, n2) {
        for (var e3, r2 = 0, o3 = s(this.props.columns); !(e3 = o3()).done; ) {
        var i2 = e3.value;
        if (r2 !== 0)
            break;
        var u2 = t3.cells[i2.index].data, a2 = n2.cells[i2.index].data;
        r2 |= typeof i2.compare == "function" ? i2.compare(u2, a2) * i2.direction : this.compare(u2, a2) * i2.direction;
        }
        return r2;
    }, o2._process = function(t3) {
        var n2 = [].concat(t3.rows);
        n2.sort(this.compareWrapper.bind(this));
        var e3 = new J(n2);
        return e3.length = t3.length, e3;
    }, n(e2, [{ key: "type", get: function() {
        return K.Sort;
    } }]), e2;
    }(Y);
    var vn = function(t2, n2, r2, o2) {
    return function(i2) {
        var u2 = i2.sort ? [].concat(i2.sort.columns) : [], s2 = u2.length, a2 = u2.find(function(n3) {
        return n3.index === t2;
        }), l2 = false, c2 = false, f2 = false, p2 = false;
        if (a2 !== void 0 ? r2 ? a2.direction === -1 ? f2 = true : p2 = true : s2 === 1 ? p2 = true : s2 > 1 && (c2 = true, l2 = true) : s2 === 0 ? l2 = true : s2 > 0 && !r2 ? (l2 = true, c2 = true) : s2 > 0 && r2 && (l2 = true), c2 && (u2 = []), l2)
        u2.push({ index: t2, direction: n2, compare: o2 });
        else if (p2) {
        var d2 = u2.indexOf(a2);
        u2[d2].direction = n2;
        } else if (f2) {
        var h2 = u2.indexOf(a2);
        u2.splice(h2, 1);
        }
        return e({}, i2, { sort: { columns: u2 } });
    };
    };
    var gn = function(t2, n2, r2) {
    return function(o2) {
        var i2 = (o2.sort ? [].concat(o2.sort.columns) : []).find(function(n3) {
        return n3.index === t2;
        });
        return e({}, o2, i2 ? vn(t2, i2.direction === 1 ? -1 : 1, n2, r2)(o2) : vn(t2, 1, n2, r2)(o2));
    };
    };
    var yn = /* @__PURE__ */ function(t2) {
    function o2() {
        return t2.apply(this, arguments) || this;
    }
    return r(o2, t2), o2.prototype._process = function(t3) {
        var n2 = {};
        return this.props.url && (n2.url = this.props.url(t3.url, this.props.columns)), this.props.body && (n2.body = this.props.body(t3.body, this.props.columns)), e({}, t3, n2);
    }, n(o2, [{ key: "type", get: function() {
        return K.ServerSort;
    } }]), o2;
    }(Y);
    function bn(t2) {
    var n2 = Et(), r2 = Lt(), o2 = vt(0), i2 = o2[0], u2 = o2[1], s2 = vt(void 0), a2 = s2[0], l2 = s2[1], c2 = jt(function(t3) {
        return t3.sort;
    }), f2 = Ht().dispatch, p2 = n2.sort;
    gt(function() {
        var t3 = d2();
        t3 && l2(t3);
    }, []), gt(function() {
        return n2.pipeline.register(a2), function() {
        return n2.pipeline.unregister(a2);
        };
    }, [n2, a2]), gt(function() {
        if (c2) {
        var n3 = c2.columns.find(function(n4) {
            return n4.index === t2.index;
        });
        u2(n3 ? n3.direction : 0);
        }
    }, [c2]), gt(function() {
        a2 && c2 && a2.setProps({ columns: c2.columns });
    }, [c2]);
    var d2 = function() {
        var t3 = K.Sort;
        return p2 && typeof p2.server == "object" && (t3 = K.ServerSort), n2.pipeline.getStepsByType(t3).length === 0 ? t3 === K.ServerSort ? new yn(e({ columns: c2 ? c2.columns : [] }, p2.server)) : new mn({ columns: c2 ? c2.columns : [] }) : null;
    };
    return w("button", { tabIndex: -1, "aria-label": r2("sort.sort" + (i2 === 1 ? "Desc" : "Asc")), title: r2("sort.sort" + (i2 === 1 ? "Desc" : "Asc")), className: et(nt("sort"), nt("sort", function(t3) {
        return t3 === 1 ? "asc" : t3 === -1 ? "desc" : "neutral";
    }(i2)), n2.className.sort), onClick: function(n3) {
        n3.preventDefault(), n3.stopPropagation(), f2(gn(t2.index, n3.shiftKey === true && p2.multiColumn, t2.compare));
    } });
    }
    function wn(t2) {
    var n2, e2 = function(t3) {
        return t3 instanceof MouseEvent ? Math.floor(t3.pageX) : Math.floor(t3.changedTouches[0].pageX);
    }, r2 = function(r3) {
        r3.stopPropagation();
        var u2, s2, a2, l2, c2, f2 = parseInt(t2.thRef.current.style.width, 10) - e2(r3);
        u2 = function(t3) {
        return o2(t3, f2);
        }, (s2 = 10) === void 0 && (s2 = 100), n2 = function() {
        var t3 = [].slice.call(arguments);
        a2 ? (clearTimeout(l2), l2 = setTimeout(function() {
            Date.now() - c2 >= s2 && (u2.apply(void 0, t3), c2 = Date.now());
        }, Math.max(s2 - (Date.now() - c2), 0))) : (u2.apply(void 0, t3), c2 = Date.now(), a2 = true);
        }, document.addEventListener("mouseup", i2), document.addEventListener("touchend", i2), document.addEventListener("mousemove", n2), document.addEventListener("touchmove", n2);
    }, o2 = function(n3, r3) {
        n3.stopPropagation();
        var o3 = t2.thRef.current;
        r3 + e2(n3) >= parseInt(o3.style.minWidth, 10) && (o3.style.width = r3 + e2(n3) + "px");
    }, i2 = function t3(e3) {
        e3.stopPropagation(), document.removeEventListener("mouseup", t3), document.removeEventListener("mousemove", n2), document.removeEventListener("touchmove", n2), document.removeEventListener("touchend", t3);
    };
    return w("div", { className: et(nt("th"), nt("resizable")), onMouseDown: r2, onTouchStart: r2, onClick: function(t3) {
        return t3.stopPropagation();
    } });
    }
    function xn(t2) {
    var n2 = Et(), r2 = yt(null), o2 = vt({}), i2 = o2[0], u2 = o2[1], s2 = Ht().dispatch;
    gt(function() {
        if (n2.fixedHeader && r2.current) {
        var t3 = r2.current.offsetTop;
        typeof t3 == "number" && u2({ top: t3 });
        }
    }, [r2]);
    var a2, l2 = function() {
        return t2.column.sort != null;
    }, c2 = function(e2) {
        e2.stopPropagation(), l2() && s2(gn(t2.index, e2.shiftKey === true && n2.sort.multiColumn, t2.column.sort.compare));
    };
    return w("th", e({ ref: r2, "data-column-id": t2.column && t2.column.id, className: et(nt("th"), l2() ? nt("th", "sort") : null, n2.fixedHeader ? nt("th", "fixed") : null, n2.className.th), onClick: c2, style: e({}, n2.style.th, { minWidth: t2.column.minWidth, width: t2.column.width }, i2, t2.style), onKeyDown: function(t3) {
        l2() && t3.which === 13 && c2(t3);
    }, rowSpan: t2.rowSpan > 1 ? t2.rowSpan : void 0, colSpan: t2.colSpan > 1 ? t2.colSpan : void 0 }, (a2 = t2.column) ? typeof a2.attributes == "function" ? a2.attributes(null, null, t2.column) : a2.attributes : {}, l2() ? { tabIndex: 0 } : {}), w("div", { className: nt("th", "content") }, t2.column.name !== void 0 ? t2.column.name : t2.column.plugin !== void 0 ? w($t, { pluginId: t2.column.plugin.id, props: { column: t2.column } }) : null), l2() && w(bn, e({ index: t2.index }, t2.column.sort)), t2.column.resizable && t2.index < n2.header.visibleColumns.length - 1 && w(wn, { column: t2.column, thRef: r2 }));
    }
    function kn() {
    var t2, n2 = Et(), e2 = jt(function(t3) {
        return t3.header;
    });
    return e2 ? w("thead", { key: e2.id, className: et(nt("thead"), n2.className.thead) }, (t2 = Gt.tabularFormat(e2.columns)).map(function(n3, r2) {
        return function(t3, n4, r3) {
        var o2 = Gt.leafColumns(e2.columns);
        return w(dn, null, t3.map(function(t4) {
            return t4.hidden ? null : function(t5, n5, e3, r4) {
            var o3 = function(t6, n6, e4) {
                var r5 = Gt.maximumDepth(t6), o4 = e4 - n6;
                return { rowSpan: Math.floor(o4 - r5 - r5 / o4), colSpan: t6.columns && t6.columns.length || 1 };
            }(t5, n5, r4);
            return w(xn, { column: t5, index: e3, colSpan: o3.colSpan, rowSpan: o3.rowSpan });
            }(t4, n4, o2.indexOf(t4), r3);
        }));
        }(n3, r2, t2.length);
    })) : null;
    }
    var Sn = function(t2) {
    return function(n2) {
        return e({}, n2, { header: t2 });
    };
    };
    function Nn() {
    var t2 = Et(), n2 = yt(null), r2 = Ht().dispatch;
    return gt(function() {
        n2 && r2(function(t3) {
        return function(n3) {
            return e({}, n3, { tableRef: t3 });
        };
        }(n2));
    }, [n2]), w("table", { ref: n2, role: "grid", className: et(nt("table"), t2.className.table), style: e({}, t2.style.table, { height: t2.height }) }, w(kn, null), w(_n, null));
    }
    function Cn() {
    var t2 = vt(true), n2 = t2[0], r2 = t2[1], o2 = yt(null), i2 = Et();
    return gt(function() {
        o2.current.children.length === 0 && r2(false);
    }, [o2]), n2 ? w("div", { ref: o2, className: et(nt("head"), i2.className.header), style: e({}, i2.style.header) }, w($t, { position: qt.Header })) : null;
    }
    function Pn() {
    var t2 = yt(null), n2 = vt(true), r2 = n2[0], o2 = n2[1], i2 = Et();
    return gt(function() {
        t2.current.children.length === 0 && o2(false);
    }, [t2]), r2 ? w("div", { ref: t2, className: et(nt("footer"), i2.className.footer), style: e({}, i2.style.footer) }, w($t, { position: qt.Footer })) : null;
    }
    function En() {
    var t2 = Et(), n2 = Ht().dispatch, r2 = jt(function(t3) {
        return t3.status;
    }), o2 = jt(function(t3) {
        return t3.data;
    }), i2 = jt(function(t3) {
        return t3.tableRef;
    }), u2 = { current: null };
    gt(function() {
        return n2(Sn(t2.header)), s2(), t2.pipeline.on("updated", s2), function() {
        return t2.pipeline.off("updated", s2);
        };
    }, []), gt(function() {
        t2.header && r2 === a.Loaded && o2 != null && o2.length && n2(Sn(t2.header.adjustWidth(t2, i2, u2)));
    }, [o2, t2, u2]);
    var s2 = function() {
        try {
        n2(function(t3) {
            return e({}, t3, { status: a.Loading });
        });
        var r3 = function(r4, o3) {
            try {
            var i3 = Promise.resolve(t2.pipeline.process()).then(function(t3) {
                n2(function(t4) {
                return function(n3) {
                    return t4 ? e({}, n3, { data: t4, status: a.Loaded }) : n3;
                };
                }(t3)), setTimeout(function() {
                n2(function(t4) {
                    return t4.status === a.Loaded ? e({}, t4, { status: a.Rendered }) : t4;
                });
                }, 0);
            });
            } catch (t3) {
            return o3(t3);
            }
            return i3 && i3.then ? i3.then(void 0, o3) : i3;
        }(0, function(t3) {
            zt.error(t3), n2(function(t4) {
            return e({}, t4, { data: null, status: a.Error });
            });
        });
        return Promise.resolve(r3 && r3.then ? r3.then(function() {
        }) : void 0);
        } catch (t3) {
        return Promise.reject(t3);
        }
    };
    return w("div", { role: "complementary", className: et("gridjs", nt("container"), r2 === a.Loading ? nt("loading") : null, t2.className.container), style: e({}, t2.style.container, { width: t2.width }) }, r2 === a.Loading && w("div", { className: nt("loading-bar") }), w(Cn, null), w("div", { className: nt("wrapper"), style: { height: t2.height } }, w(Nn, null)), w(Pn, null), w("div", { ref: u2, id: "gridjs-temp", className: nt("temp") }));
    }
    var In = /* @__PURE__ */ function(t2) {
    function n2(n3) {
        var e3;
        return (e3 = t2.call(this) || this).config = void 0, e3.plugin = void 0, e3.config = new fn().assign({ instance: i(e3), eventEmitter: i(e3) }).update(n3), e3.plugin = e3.config.plugin, e3;
    }
    r(n2, t2);
    var e2 = n2.prototype;
    return e2.updateConfig = function(t3) {
        return this.config.update(t3), this;
    }, e2.createElement = function() {
        return w(cn.Provider, { value: this.config, children: w(En, {}) });
    }, e2.forceRender = function() {
        return this.config && this.config.container || zt.error("Container is empty. Make sure you call render() before forceRender()", true), this.destroy(), q(this.createElement(), this.config.container), this;
    }, e2.destroy = function() {
        this.config.pipeline.clearCache(), q(null, this.config.container);
    }, e2.render = function(t3) {
        return t3 || zt.error("Container element cannot be null", true), t3.childNodes.length > 0 ? (zt.error("The container element " + t3 + " is not empty. Make sure the container is empty and call render() again"), this) : (this.config.container = t3, q(this.createElement(), t3), this);
    }, n2;
    }(Q);
    var gridjs_default = null;

    // main/test.js
    return gridjs_6_0_exports;
}
const gridjs = await getGridJs()
export {
  test_default as default
};
