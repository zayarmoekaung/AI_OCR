export interface Item {
    name: string;
    price: number;
    quantity: number;
}
export interface Receipt {
    merchant: string;
    date: string;
    total: number;
    items: Item[];
}