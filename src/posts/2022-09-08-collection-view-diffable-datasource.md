---
layout: post
title: Collection View With Diffable Datasource
slug: collectionview_diffable_datasource
date: 2022-09-09 23:57
status: publish
author: walker
categories: 
  - iOS
tags:
  - CollectionView
  - Diffable Datasource
  - Compositional Layout
  - Snapshot
---

这篇文章有
* collection view自定义布局的一些心得体会和查阅文档时的一些笔记
* Compositional layout笔记 （少量）
* diffable datasource笔记


# Compositional Layout

* Group 宽高给够（或estimate），Item固定大小，就成了一个FlowLayout
* 设定section垂直方向行为为滚动(分页，靠边等），则不会折行
  * `.continuousGroupLeadingBoundary` 的意思是如果一行摆不下，正常情况下会折行，这一行后面就会剩下空白，当你做成continous后，下一个元素也会排在空白后，而不是直接就接在后面了
  * `.paging`和`.groupPageing`的区别则是一次滚动一页还是一个group

# Diffable Data Sources

* A *diffable data source* stores a list of section and item *identifiers*
  *  In contrast, a custom data source that conforms to [`UICollectionViewDataSource`](https://developer.apple.com/documentation/uikit/uicollectionviewdatasource) uses *indices* and *index paths*, which aren’t stable. 
    * They represent the **location** of sections and items, which can change as the data source adds, removes, and rearranges the contents of a collection view.
    * 相反Diffable Data Source却能根据identifier追溯到其location
* To use a value as an identifier, its data type must conform to the [`Hashable`](https://developer.apple.com/documentation/swift/hashable) protocol. 
  * Hashing能让集合成为“键”，提供快速lookup能力
    * 比如set, dictionary, snapshot
  *  can determine the differences between its **current** snapshot and **another** snapshot. 

### Define the Diffable Data Source

```swift
@preconcurrency @MainActor class UICollectionViewDiffableDataSource<SectionIdentifierType, ItemIdentifierType> : NSObject where SectionIdentifierType : Hashable, SectionIdentifierType : Sendable, ItemIdentifierType : Hashable, ItemIdentifierType : Sendable

// 声明示例
private var recipeListDataSource: UICollectionViewDiffableDataSource<RecipeListSection, Recipe.ID>!

private enum RecipeListSection: Int {
    case main
}

struct Recipe: Identifiable, Codable {
    var id: Int
    var title: String
    var prepTime: Int   // In seconds.
    var cookTime: Int   // In seconds.
    var servings: String
    var ingredients: String
    var directions: String
    var isFavorite: Bool
    var collections: [String]
    fileprivate var addedOn: Date? = Date()
    fileprivate var imageNames: [String]
}
```

1. section是枚举，枚举就是正整数
2. Recipe conforming to `Identifiable`，automatically exposes the associated type [`ID`](https://developer.apple.com/documentation/swift/identifiable/id-swift.associatedtype)
3. 整个`Recipe`结构体不必是`Hashable`的，因为存在Datasource和Snapshot里的仅仅只是`identifiers`
   1. Using the `Recipe.ID` as the item identifier type for the `recipeListDataSource` means that the **data source**, and any **snapshots** applied to it, **contains only** `Recipe.ID` values and not the complete recipe data. 

### Configure the Diffable Data Source

```swift
// Create a cell registration that the diffable data source will use.
let recipeCellRegistration = UICollectionView.CellRegistration<UICollectionViewListCell, Recipe> { cell, indexPath, recipe in
    // 会带着cell对象，位置和应的数据源数据来请求配置当前cell 
    // 这里进行了两种配置，
    // 1. 一种是对contentConfiguration进行配置（应该就是包了一层，没对cell暴露出来的subview直接进行设置）
    var contentConfiguration = UIListContentConfiguration.subtitleCell()
    contentConfiguration.text = recipe.title
    contentConfiguration.secondaryText = recipe.subtitle
    contentConfiguration.image = recipe.smallImage
    contentConfiguration.imageProperties.cornerRadius = 4
    contentConfiguration.imageProperties.maximumSize = CGSize(width: 60, height: 60)
    
    cell.contentConfiguration = contentConfiguration
    
    // 2. 这里就是直接对cell的subview来进行设置了，所以理论上上一节的内容应该也可以直接对cell来配置
    if recipe.isFavorite {
        let image = UIImage(systemName: "heart.fill")
        let accessoryConfiguration = UICellAccessory.CustomViewConfiguration(customView: UIImageView(image: image), placement: .trailing(displayed: .always), cell.accessories = [.customView(configuration: accessoryConfiguration)]
    } else {
        cell.accessories = []
    }
}

// Create the diffable data source and its cell provider.
recipeListDataSource = UICollectionViewDiffableDataSource(collectionView: collectionView) {
    collectionView, indexPath, identifier -> UICollectionViewCell in
    // `identifier` is an instance of `Recipe.ID`. Use it to
    // retrieve the recipe from the backing data store.
    let recipe = dataStore.recipe(with: identifier)!
    // 这里既是传入注册cell的方法的地方，也是那个方法的handler里三个参数的来源
    return collectionView.dequeueConfiguredReusableCell(using: recipeCellRegistration, for: indexPath, item: recipe)
}
```

* The `configureDataSource()` method creates a cell **registration** and provides a handler closure that **configures each cell** with data from a recipe. 

### Load the Diffable Data Source with Identifiers

```swift
private func loadRecipeData() {
    // Retrieve the list of recipe identifiers determined based on a
    // selected sidebar item such as All Recipes or Favorites.
    guard let recipeIds = recipeSplitViewController.selectedRecipes?.recipeIds()
    else { return }
    
    // Update the collection view by adding the recipe identifiers to
    // a new snapshot, and apply the snapshot to the diffable data source.
    var snapshot = NSDiffableDataSourceSnapshot<RecipeListSection, Recipe.ID>()
    snapshot.appendSections([.main])
    snapshot.appendItems(recipeIds, toSection: .main)
    recipeListDataSource.applySnapshotUsingReloadData(snapshot) // 初始化用这个，reload代表完全重设
    // 更新的话用 apply(_:animatingDifferences:) 这样有动画
}
```

### Insert, Delete, and Move Items

* To **handle changes** to a data collection, the app **creates a new snapshot** that represents the current state of the data collection and **applies** it to the diffable data source. 
* The data source **compares** its current snapshot with the new snapshot to **determine the changes**. 
* Then it performs the necessary inserts, deletes, and moves into the collection view based on those changes.

```swift
var snapshot = NSDiffableDataSourceSnapshot<RecipeListSection, Recipe.ID>()
snapshot.appendSections([.main]) // section是直接重建的，而不是从哪去retrieve一个, 因为它代表的是ID，只要值一致就行
snapshot.appendItems(selectedRecipeIds, toSection: .main) // 这里是.main的全量数据，即增删后的结果集
recipeListDataSource.apply(snapshot, animatingDifferences: true)
```

* 增删其实就是新建一个snapshot，datasource会根据identifiers来比较哪些多了哪些少了。
  * 因为只比较“数量“，所以只要用这些id去新建snapshot就可以了，不存在把旧的**retrieve**出来

### Update Existing Items

* To handle changes to the properties of an **EXISTING** item, an app retrieves the **current snapshot** from the diffable data source and calls either [`reconfigureItems(_:)`](https://developer.apple.com/documentation/uikit/nsdiffabledatasourcesnapshot/3804468-reconfigureitems) or [`reloadItems(_:)`](https://developer.apple.com/documentation/uikit/nsdiffabledatasourcesnapshot/3375783-reloaditems) on the snapshot.  -> then `Apply` to snapshot

```swift
var snapshot = recipeListDataSource.snapshot()  // 这次是retrieve了
// Update the recipe's data displayed in the collection view.
snapshot.reconfigureItems([recipeId]) // 传入identifier
recipeListDataSource.apply(snapshot, animatingDifferences: true)
```

*  the data source invokes its cell provider closure, 

### Populate Snapshots with Lightweight Data Structures

* 对整个item对象做Hash，适用于快速建模，或数据源不会变更的场景（比如菜单）。
  * 因为item对象的任何属性变化都会被认为有过改动导致重绘，也会产生一些副作用，比如重绘之前的状态都会被清掉（如selected）
* 实践中，不会对设置datasource的时候专门给个identifier集合，而数据源用别的集合，每次都是用identifier从集合里找item这种方式，而是重写item的hash方法和equal方法，让其只观察id字段

### NSDiffableDataSourceSnapshot

* A representation of **the state of the data** in a `view` at a **specific point in time**.
* Diffable data sources use *snapshots* to provide data for collection views and table views. 
* You use a snapshot to set up the **initial state** of the data that a view displays, and you use snapshots to reflect **changes to the data** that the view displays.
* The data in a snapshot is made up of the **sections** and **items**
  * Each of your sections and items must have unique identifiers that conform to the [`Hashable`](https://developer.apple.com/documentation/swift/hashable) protocol. 

```swift
// Create a snapshot.
var snapshot = NSDiffableDataSourceSnapshot<Int, UUID>()        

// Populate the snapshot.
snapshot.appendSections([0])
snapshot.appendItems([UUID(), UUID(), UUID()])

// Apply the snapshot.
dataSource.apply(snapshot, animatingDifferences: true)
```



## NSDiffableDataSourceSectionSnapshot

* A representation of **the state of the data** in a `layout section` at a specific point in time. 
  * 注意与`dataSourceSnapshot`定义的区别

* A section snapshot represents the data for a single section in a collection view or table view. 
* Through a section snapshot, you set up the **initial state** of the data that displays in an individual section of your view, and later **update that data**.
* You can use section snapshots **with** or **instead** of an [`NSDiffableDataSourceSnapshot`](https://developer.apple.com/documentation/uikit/nsdiffabledatasourcesnapshot)
*  Use a section snapshot when you need precise management of the data in a section of your layout
  * such as when the sections of your layout acquire their data from **different sources**. 
  * 不同的section来自不同的数据源的话，倾向于用sectionSnapshot


```swift
for section in Section.allCases {
    // Create a section snapshot
    var sectionSnapshot = NSDiffableDataSourceSectionSnapshot<String>()
    
    // Populate the section snapshot
    sectionSnapshot.append(["Food", "Drinks"])
    sectionSnapshot.append(["🍏", "🍓", "🥐"], to: "Food")
    
    // Apply the section snapshot
    dataSource.apply(sectionSnapshot,
                     to: section,
                     animatingDifferences: true)
}
```

# 苹果CollectionView教程文档

**The Layout Object Controls the Visual Presentation**

* The layout object is solely responsible for determining the **placement and visual styling** of items within the collection view
*  do not confuse what a layout object does with the `layoutSubviews` method used to reposition child views inside a parent view.
  * A layout object **never** touches the views it manages directly because it **does not actually own** any of those views. 
  *  it generates attributes that describe the location, size, and visual appearance of the cells, supplementary views, and decoration views in the collection view.
  * It is then the job of the collection view to apply those attributes to the actual view objects.
  * 这就是需要提供两个代理方法的原因，一个提供view，一个提供布局配置

**Transitioning Between Layouts**

* The easiest way to transition between layouts is by using the `setCollectionViewLayout:animated:` method. 
* However, if you require control of the transition or want it to be interactive, use a `UICollectionViewTransitionLayout` object.
* The `UICollectionViewTransitionLayout` class is a special type of layout that gets installed as the collection view’s layout object when transitioning to a new layout. 
  * With a transition layout object, you can have objects follow a **non linear** path, use a different **timing algorithm**, or move according to incoming touch events. 
* The `UICollectionViewLayout` class provides **several methods** for **tracking** the transition between layouts. 
* `UICollectionViewTransitionLayout` objects track the completion of a transition through the `transitionProgress` property. 
* As the transition occurs, your code updates this property **periodically** to indicate the completion percentage of the transition. 

通用流程：

1. Create an instance of the standard class or your own custom class using the `initWithCurrentLayout:nextLayout:` method.
2. Communicate the progress of the transition by periodically modifying the `transitionProgress` property. Do not forget to invalidate the layout using the collection view’s `invalidateLayout` method after changing the transition’s progress.
3. Implement the `collectionView:transitionLayoutForOldLayout:newLayout:` method in your collection view’s delegate and return your transition layout object.
4. Optionally modify values for your layout using the `updateValue:forAnimatedKey:` method to indicate changed values relevant to your layout object. The stable value in this case is 0.

**Customizing the Flow Layout Attributes**

* Flowlayout在一条线上排列元素，到达了边界就换行，新起一条线
* 元素大小可以通过`itemSize` 属性设置，如果大小不同，则通过`[collectionView:layout:sizeForItemAtIndexPath:](https://developer.apple.com/documentation/uikit/uicollectionviewdelegateflowlayout/1617708-collectionview)`代理方法设置
* 但是，同一行上不同的高度的cell会垂直居中排列，这点要注意
* `minimum spacing`设置的只是同一行元素的“最小间距”，如果布局的时候一行下一个元素放不下了，但是剩余的空间很多，这个一行的元素间距会拉大
  * 行间距同理，根据上一条描述，元素是垂直居中排列的，所以最小行间距设置的是上下两行间最高的元素的距离
